"""Declarative audio benchmark definitions.

Language availability follows what actually exists on the Hub:

  kk + ru + en : sakura, wavcaps, wavcaps_qa, asr
  kk only      : spokenmqa

Row counts were checked on the Hub and match exactly across languages: SAKURA
500 per category, WavCaps 1730, WavCapsQA 304.  SAKURA aligns on its ``file``
column; WavCaps and WavCapsQA are positionally parallel (verified at both ends
of the file).  ASR is FLEURS, where each language is its own recording set, so
it is deliberately unaligned.

S2TT is intentionally absent.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from core.parsing import parse_string_list
from core.registry import BenchmarkSpec, Registry, Sample, Source, TaskType

REGISTRY = Registry()
register = REGISTRY.register

Row = Dict[str, Any]

AUDIO_COLUMNS = ["audio", "context"]

# SAKURA ships one config per attribute; the short name is what the summary's
# per-category breakdown reports.
SAKURA_CATEGORIES = {
    "AnimalQA": "animal",
    "LanguageQA": "language",
    "EmotionQA": "emotion",
    "GenderQA": "gender",
}

SPOKENMQA_SUBSETS = ["long_digit", "short_digit",
                     "single_step_reasoning", "multi_step_reasoning"]

# The English originals are one repo per subset (split "english") rather than
# one repo with configs, and multi-step is split across two repos that the
# reference implementation concatenates. group_map folds those two back into a
# single cell so the sub-metrics line up with the Kazakh set.
SPOKENMQA_EN_CONFIGS = ["long_digit", "short_digit", "single_step_reasoning",
                        "multi_step_reasoning_1", "multi_step_reasoning_2"]
SPOKENMQA_EN_GROUPS = {"multi_step_reasoning_1": "multi_step_reasoning",
                       "multi_step_reasoning_2": "multi_step_reasoning"}

FLEURS_CONFIGS = {"kk": "kk_kz", "ru": "ru_ru", "en": "en_us"}

# "(b) cat" / "b)" / "B" -> "B"
_CHOICE = re.compile(r"^\s*\(?\s*([a-zA-Z])\s*[\)\.\:]?\s")


def _choice_letter(raw: Any) -> Optional[str]:
    """The option letter from SAKURA's '(b) cat' style answers."""
    text = str(raw or "").strip()
    if not text:
        return None
    match = _CHOICE.match(text + " ")
    if match:
        return match.group(1).upper()
    stripped = text.strip("() .:")
    return stripped.upper() if len(stripped) == 1 and stripped.isalpha() else None


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

def adapt_sakura(row: Row, idx: int) -> Optional[List[Sample]]:
    """One SAKURA row carries both a single-hop and a multi-hop question.

    Each becomes its own evaluation item, grouped as e.g. ``animal_single`` so
    the summary reports the eight cells the benchmark is normally read as.
    """
    config = str(row.get("__config__") or "")
    category = SAKURA_CATEGORIES.get(config, config or "unknown")
    # `file` is shared by the translation and the English original, which is
    # what makes the three languages line up item for item.
    base = f"{category}/{row.get('file') or idx}"

    samples: List[Sample] = []
    for hop in ("single", "multi"):
        instruction = row.get(f"{hop}_instruction")
        gold = _choice_letter(row.get(f"{hop}_answer"))
        if not instruction or gold is None:
            continue
        sample = Sample(
            uid=f"{base}:{hop}",
            question=str(instruction),
            reference=gold,
            group=f"{category}_{hop}",
        )
        sample.meta["hop"] = hop
        sample.meta["category"] = category
        if row.get("attribute_label"):
            sample.meta["attribute_label"] = str(row["attribute_label"])
        samples.append(sample)
    return samples or None


def adapt_spokenmqa(row: Row, idx: int) -> Optional[Sample]:
    answer = row.get("answer")
    if answer in (None, ""):
        return None
    # Gold answers may carry a GSM8K-style '#### x' tail.
    text = str(answer)
    if "####" in text:
        text = text.split("####")[-1]
    text = text.strip().replace(",", "")
    try:
        reference = float(text)
    except ValueError:
        return None

    sample = Sample(uid=str(row.get("file") or idx),
                    question=str(row.get("instruction", "")),
                    reference=reference,
                    group=str(row.get("__config__") or "unknown"))
    if row.get("audio_transcription"):
        sample.meta["audio_transcription"] = str(row["audio_transcription"])
    return sample


def adapt_wavcaps(row: Row, idx: int) -> Optional[Sample]:
    """WavCaps and WavCapsQA share a schema: context (audio), instruction, answer."""
    reference = row.get("answer")
    if reference in (None, ""):
        return None
    return Sample(uid=str(idx), question=str(row.get("instruction", "")),
                  reference=str(reference))


def adapt_fleurs(row: Row, idx: int) -> Optional[Sample]:
    """FLEURS from the Hub and a local manifest share this adapter; the local
    reader normalises whatever text column it found into ``text``."""
    reference = (row.get("raw_transcription") or row.get("transcription")
                 or row.get("text"))
    if not reference:
        return None
    sample = Sample(uid=str(row.get("id") or row.get("file_name") or idx),
                    question="",
                    reference=str(reference),
                    group=str(row.get("__config__") or "unknown"))
    if row.get("transcription"):
        sample.meta["transcription"] = str(row["transcription"])
    return sample


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

register(BenchmarkSpec(
    key="sakura",
    task=TaskType.MCQ,
    description="SAKURA audio MCQ, single- and multi-hop, over four attributes.",
    align=True,
    max_options=4,
    group_name="category_hop",
    print_groups=True,
    sources={
        "kk": Source("issai/SAKURA_Kazakh_Russian", adapt_sakura,
                     configs=list(SAKURA_CATEGORIES), split="kazakh",
                     public=False, audio_columns=AUDIO_COLUMNS),
        "ru": Source("issai/SAKURA_Kazakh_Russian", adapt_sakura,
                     configs=list(SAKURA_CATEGORIES), split="russian",
                     public=False, audio_columns=AUDIO_COLUMNS),
        # The English original is published as four sibling repos, one per
        # attribute; the "{config}" placeholder expands to each in turn.
        "en": Source("SLLM-multi-hop/{config}", adapt_sakura,
                     configs=list(SAKURA_CATEGORIES), split="test",
                     audio_columns=AUDIO_COLUMNS,
                     note="four sibling repos, joined on category + audio file name"),
    },
))

register(BenchmarkSpec(
    key="spokenmqa",
    task=TaskType.MATH_NUMERIC,
    description="SpokenMQA spoken math — digits and single/multi-step reasoning.",
    group_name="subset",
    print_groups=True,
    sources={
        "kk": Source("issai/SpokenMQA_Kazakh", adapt_spokenmqa,
                     configs=SPOKENMQA_SUBSETS, split="kazakh", public=False,
                     audio_columns=AUDIO_COLUMNS),
        "en": Source("issai/{config}", adapt_spokenmqa,
                     configs=SPOKENMQA_EN_CONFIGS, split="english",
                     public=False, audio_columns=AUDIO_COLUMNS,
                     group_map=SPOKENMQA_EN_GROUPS,
                     note="one repo per subset; multi-step merged from two"),
    },
))

register(BenchmarkSpec(
    key="wavcaps",
    task=TaskType.AUDIO_JUDGE,
    description="WavCaps audio captioning — free-form, judged 0/1.",
    align=True,
    sources={
        "kk": Source("issai/WavCaps_Kazakh_Russian", adapt_wavcaps,
                     config="default", split="kazakh", public=False,
                     audio_columns=AUDIO_COLUMNS),
        "ru": Source("issai/WavCaps_Kazakh_Russian", adapt_wavcaps,
                     config="default", split="russian", public=False,
                     audio_columns=AUDIO_COLUMNS),
        "en": Source("AudioLLMs/wavcaps_test", adapt_wavcaps, config="default",
                     split="test", audio_columns=AUDIO_COLUMNS,
                     note="1730 items, positionally parallel with the translation"),
    },
))

register(BenchmarkSpec(
    key="wavcaps_qa",
    task=TaskType.AUDIO_JUDGE,
    description="WavCaps QA — question answering over audio, judged 0/1.",
    align=True,
    sources={
        "kk": Source("issai/WavCapsQA_Kazakh_Russian", adapt_wavcaps,
                     config="default", split="kazakh", public=False,
                     audio_columns=AUDIO_COLUMNS),
        "ru": Source("issai/WavCapsQA_Kazakh_Russian", adapt_wavcaps,
                     config="default", split="russian", public=False,
                     audio_columns=AUDIO_COLUMNS),
        "en": Source("AudioLLMs/wavcaps_qa_test", adapt_wavcaps,
                     config="default", split="test", audio_columns=AUDIO_COLUMNS,
                     note="304 items, positionally parallel with the translation"),
    },
))

# A FLEURS download names its transcription files by ISO-639-3 code and pairs
# source with target; ASR is the src == tgt file. These are tried first, then
# the generic per-language-subdirectory search.
FLEURS_MANIFESTS = {
    "kk": ["val_test_kaz_kaz.tsv", "kaz_kaz.tsv", "kk_kz/*.tsv",
           "kk_kz/*.csv", "kk_kz/*.jsonl", "kk/*.tsv", "kaz/*.tsv"],
    "ru": ["val_test_rus_rus.tsv", "rus_rus.tsv", "ru_ru/*.tsv",
           "ru_ru/*.csv", "ru_ru/*.jsonl", "ru/*.tsv", "rus/*.tsv"],
    "en": ["val_test_eng_eng.tsv", "eng_eng.tsv", "en_us/*.tsv",
           "en_us/*.csv", "en_us/*.jsonl", "en/*.tsv", "eng/*.tsv"],
}


def _asr_source(lang: str, hub_note: str) -> Source:
    """The user's own ASR corpus when ASR_DATA_DIR points at one, else FLEURS.

    A FLEURS export is read as-is: ``<dir>/val_test_kaz_kaz.tsv`` with its
    ``id / audio / text`` header and audio paths like ``wavs/test_kk/x.wav``.
    Other layouts work too — ``<dir>/kk_kz/test.tsv`` and friends — and the
    manifest may be TSV, CSV or JSONL with the audio column named audio /
    file_name / path / file and the text column text / transcription /
    raw_transcription / sentence.
    """
    config = FLEURS_CONFIGS[lang]
    return Source(
        path="", adapter=adapt_fleurs, kind="manifest", config=config,
        dir_key="asr", manifest_patterns=FLEURS_MANIFESTS[lang],
        note="local ASR corpus (ASR_DATA_DIR)",
        fallback=Source("google/fleurs", adapt_fleurs, config=config,
                        split="test", audio_columns=["audio"], note=hub_note),
    )


register(BenchmarkSpec(
    key="asr",
    task=TaskType.ASR_WER,
    description="Speech recognition — word error rate (lower is better).",
    align=False,
    group_name="corpus",
    sources={
        "kk": _asr_source("kk", "FLEURS kk_kz test, 856 utterances"),
        "ru": _asr_source("ru", "FLEURS ru_ru test, 775 utterances"),
        "en": _asr_source("en", "FLEURS en_us test, 647 utterances; each "
                                "language is its own recording set, so these "
                                "are not parallel"),
    },
))
