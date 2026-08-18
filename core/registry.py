"""Core data model shared by every modality.

A benchmark is declared once, declaratively, as a :class:`BenchmarkSpec`
holding one :class:`Source` per available language.  Adding a benchmark means
adding a spec in a modality's ``specs.py`` — no new runner code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


class TaskType(str, Enum):
    """Determines prompt shape, extraction and scoring."""

    # --- text
    MCQ = "mcq"                      # letter-labelled multiple choice
    MATH_NUMERIC = "math_numeric"    # single numeric answer, exact match
    MATH_SYMBOLIC = "math_symbolic"  # LaTeX/symbolic answer, sympy-verified
    SPELLING = "spelling"            # sentence correction, exact match
    QA_JUDGE = "qa_judge"            # extractive QA, judged 0/1
    RAG_JUDGE = "rag_judge"          # grounded generation, judged 0/1
    IF_JUDGE = "if_judge"            # constraint following, judged 0/1

    # --- vision
    FLEXIBLE = "flexible"            # answer is a letter OR a value
    MATH_MIXED = "math_mixed"        # MCQ when choices exist, else symbolic
    OCR_MATCH = "ocr_match"          # official OCRBench containment rule
    OCR_JUDGE = "ocr_judge"          # OCR text, judged 0/1
    BABY_MIXED = "baby_mixed"        # choice items exact, blank items judged

    # --- audio
    AUDIO_JUDGE = "audio_judge"      # captioning / audio QA, judged 0/1
    ASR_WER = "asr_wer"              # transcription, scored by word error rate

    # --- safety
    SAFETY_RUBRIC = "safety_rubric"  # judged against a per-risk-area rubric

    # --- translation
    TRANSLATION = "translation"      # scored by a neural MT metric (XCOMET)

    @property
    def is_judged(self) -> bool:
        """True when some or all items need the batch LLM judge."""
        return self in (
            TaskType.QA_JUDGE, TaskType.RAG_JUDGE, TaskType.IF_JUDGE,
            TaskType.OCR_JUDGE, TaskType.BABY_MIXED, TaskType.AUDIO_JUDGE,
            TaskType.SAFETY_RUBRIC,
        )

    @property
    def lower_is_better(self) -> bool:
        return self is TaskType.ASR_WER


@dataclass
class Sample:
    """A single normalised evaluation item, language-independent in shape.

    ``uid`` is the alignment key.  When a benchmark sets ``align=True`` the same
    ``uid`` must denote the same underlying item in every language, so that the
    kk / ru / en runs cover an identical item set.
    """

    uid: str
    question: str = ""
    options: Optional[List[str]] = None
    context: Optional[str] = None
    documents: Optional[str] = None
    constraints: Optional[str] = None
    hint: Optional[str] = None

    # Ground truth. MCQ -> letter; MATH_NUMERIC -> float; others -> str.
    reference: Any = None

    # Fetched through a thunk, so only in-flight items hold pixels/samples.
    image_loader: Optional[Callable[[], Any]] = None
    audio_loader: Optional[Callable[[], Any]] = None

    # Free-form extras kept in the output records (never prompted).
    meta: Dict[str, Any] = field(default_factory=dict)
    # Optional grouping key for per-subject / per-category sub-metrics.
    group: Optional[str] = None

    @property
    def has_image(self) -> bool:
        return self.image_loader is not None

    @property
    def has_audio(self) -> bool:
        return self.audio_loader is not None


# Turns one raw row into a Sample, a list of Samples (when a row carries
# several items, as SAKURA does), or None to drop it. The row also carries
# `__config__` and `__split__` so an adapter knows which subset it is reading.
AdapterResult = Union[None, "Sample", List["Sample"]]
Adapter = Callable[[Dict[str, Any], int], AdapterResult]


@dataclass
class Source:
    """Where one language's data for a benchmark comes from."""

    path: str                                   # HF repo id, or local path
    adapter: Adapter
    # "hf"       - a dataset on the Hub
    # "jsonl"    - a local JSONL file under the data directory
    # "manifest" - a local TSV/CSV/JSONL manifest plus its audio files
    kind: str = "hf"
    config: Optional[str] = None
    configs: Optional[List[str]] = None         # concatenated in order
    split: str = "test"
    splits: Optional[List[str]] = None          # concatenated in order
    public: bool = True                         # False -> send HF_TOKEN
    # Some HF repos expose language variants as configs discovered at runtime.
    configs_from_hub: bool = False
    # Append a per-base counter to each uid, for datasets with no item id
    # where alignment rests on position (MMLU: "abstract_algebra#0", ...).
    uid_counter: bool = False
    # Rewrites group labels so sub-metrics line up across languages.
    group_map: Optional[Dict[str, str]] = None
    # Candidate media columns, first present wins; loaded lazily.
    image_columns: Optional[List[str]] = None
    audio_columns: Optional[List[str]] = None
    # Resampled on decode, so the model sees one rate.
    audio_sampling_rate: int = 16000
    # Overrides BenchmarkSpec.task for this language alone (OCRBench).
    task: Optional[TaskType] = None
    # Globs tried before the generic manifest search.
    manifest_patterns: Optional[List[str]] = None
    # Which configured directory a local path resolves against.
    dir_key: str = "data"
    # Used when this source yields nothing (ASR falls back to the Hub).
    fallback: Optional["Source"] = None
    # Note surfaced in logs/summary, e.g. how an alignment is established.
    note: str = ""

    def parts(self) -> List[tuple]:
        """Every (config, split) pair this source is made of, in order."""
        configs = list(self.configs) if self.configs else [self.config]
        splits = list(self.splits) if self.splits else [self.split]
        return [(cfg, split) for cfg in configs for split in splits]


@dataclass
class BenchmarkSpec:
    """A benchmark across all the languages it exists in."""

    key: str
    task: TaskType
    sources: Dict[str, Source]                  # language code -> Source
    description: str = ""

    # Intersect uids across languages so every language scores the same items.
    align: bool = False
    # Number of answer options when fixed (guards letter extraction).
    max_options: Optional[int] = None
    # Sub-metric grouping label, e.g. "subject" or "category".
    group_name: Optional[str] = None
    # Also print the per-group breakdown in the terminal. On for SAKURA and
    # SpokenMQA; off for MMLU-style benchmarks with dozens of subjects.
    print_groups: bool = False

    @property
    def languages(self) -> List[str]:
        return [lang for lang in ("kk", "ru", "en") if lang in self.sources]

    def task_for(self, lang: str) -> TaskType:
        """The task type for one language — a Source may override the spec."""
        source = self.sources.get(lang)
        if source is not None and source.task is not None:
            return source.task
        return self.task


class Registry(dict):
    """An ordered name -> BenchmarkSpec map, one per modality."""

    def register(self, spec: BenchmarkSpec) -> BenchmarkSpec:
        if spec.key in self:
            raise ValueError(f"Duplicate benchmark key: {spec.key}")
        self[spec.key] = spec
        return spec

    def resolve(self, names: List[str],
                skip_unknown: bool = False) -> List[BenchmarkSpec]:
        """Expand a user selection into specs, preserving registry order.

        ``skip_unknown`` is for running every modality in one go: a key names
        exactly one modality, so the other two must ignore it rather than fail.
        """
        if not names or "all" in names:
            return list(self.values())
        unknown = [n for n in names if n not in self]
        if unknown and not skip_unknown:
            raise ValueError(
                f"Unknown benchmark(s): {', '.join(unknown)}. "
                f"Available: {', '.join(self)}"
            )
        return [self[n] for n in names if n in self]
