"""Declarative vision benchmark definitions.

Language availability follows what actually exists on the Hub:

  kk + ru + en : realworldqa, mmstar
  kk + en      : ai2d, mathvista, mathvision, mmbench, ocrbench, babyvision
  (no Russian vision sets beyond RealWorldQA and MMStar)

The English side comes from the upstream originals.  Every pairing below was
checked row-by-row against the Hub at both ends of the file: realworldqa,
mmstar, ai2d, mathvista and babyvision are positionally parallel with the
Kazakh translation, mathvision joins on its item id, and MMBench is *not*
parallel (the Kazakh set was re-indexed from zero), so it is left unaligned.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from core.parsing import LETTERS, parse_string_list
from core.registry import BenchmarkSpec, Registry, Sample, Source, TaskType

REGISTRY = Registry()
register = REGISTRY.register

Row = Dict[str, Any]

# Every vision source is probed for these columns, first match wins. MathVista
# and MathVision keep the path in `image` and the pixels in `decoded_image`.
IMAGE_COLUMNS = ["decoded_image", "image"]

_IMAGE_TAG = re.compile(r"<image\s*\d*>")
# MMBench stores unused option cells as the literal string "nan".
# "none"/"null" are NOT included: they can be genuine answers.
_NAN = {"", "nan"}


def _clean(value: Any) -> Optional[str]:
    """Normalise a cell that may legitimately be the string 'nan'."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in _NAN else text


def _strip_image_tags(question: Any) -> str:
    return _IMAGE_TAG.sub("", str(question or "")).strip()


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

def adapt_realworldqa(row: Row, idx: int) -> Optional[Sample]:
    """Options are embedded in the question text; the answer may be a letter
    or a word, so this is scored flexibly."""
    reference = _clean(row.get("answer") or row.get("Answer"))
    if reference is None:
        return None
    return Sample(uid=str(idx), question=str(row.get("question", "")),
                  reference=reference)


def adapt_mmstar(row: Row, idx: int) -> Optional[Sample]:
    """MMStar also embeds its labelled options in the question text."""
    reference = _clean(row.get("answer"))
    if reference is None:
        return None
    return Sample(uid=str(idx), question=str(row.get("question", "")),
                  reference=reference.upper(),
                  group=str(row.get("category", "unknown")))


def adapt_ai2d(row: Row, idx: int) -> Optional[Sample]:
    options = parse_string_list(row.get("options"))
    if not options:
        return None
    try:
        gold = int(row["answer"])          # 0-indexed
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 <= gold < len(options)):
        return None
    return Sample(uid=str(idx), question=str(row.get("question", "")),
                  options=options, reference=LETTERS[gold])


def _adapt_math_mixed(row: Row, idx: int, uid: str, question: Any,
                      choices_key: str) -> Optional[Sample]:
    """MathVista / MathVision: multiple choice when options exist, else the
    answer is an open-ended value graded symbolically."""
    reference = _clean(row.get("answer"))
    if reference is None:
        return None
    options = parse_string_list(row.get(choices_key)) or []
    sample = Sample(
        uid=uid,
        question=_strip_image_tags(question),
        options=options or None,
        reference=reference,
        group="mcq" if options else "open",
    )
    precision = row.get("precision")
    if precision not in (None, ""):
        try:
            sample.meta["precision"] = int(float(precision))
        except (TypeError, ValueError):
            pass
    return sample


def adapt_mathvista(row: Row, idx: int) -> Optional[Sample]:
    return _adapt_math_mixed(row, idx, str(idx), row.get("question"), "choices")


def adapt_mathvision(row: Row, idx: int) -> Optional[Sample]:
    # The Kazakh set calls it `index`, the English original calls it `id`;
    # both hold the same value for the same problem.
    uid = str(row.get("index", row.get("id", idx)))
    sample = _adapt_math_mixed(row, idx, uid, row.get("question"), "options")
    if sample is not None and row.get("level") not in (None, ""):
        sample.meta["level"] = row.get("level")
    return sample


def adapt_mmbench(row: Row, idx: int) -> Optional[Sample]:
    """A/B/C/D live in separate columns and unused ones hold the string 'nan'."""
    cells = [_clean(row.get(letter)) for letter in ("A", "B", "C", "D")]
    options: List[str] = []
    for cell in cells:
        if cell is None:
            break          # trailing unused options; an interior gap is invalid
        options.append(cell)
    if len(options) < 2 or any(c is not None for c in cells[len(options):]):
        return None

    reference = _clean(row.get("answer"))
    if reference is None or reference.upper() not in LETTERS[:len(options)]:
        return None

    return Sample(uid=str(row.get("index", idx)),
                  question=str(row.get("question", "")),
                  options=options,
                  reference=reference.upper(),
                  hint=_clean(row.get("hint")),
                  group=str(row.get("category", "unknown")))


def adapt_ocrbench(row: Row, idx: int) -> Optional[Sample]:
    """The English original allows several acceptable strings per item."""
    raw = row.get("answer")
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        references = [str(a) for a in raw if _clean(a)]
    else:
        parsed = parse_string_list(raw) if str(raw).startswith("[") else None
        references = parsed if parsed else [str(raw)]
    references = [r for r in references if r.strip()]
    if not references:
        return None

    sample = Sample(uid=str(idx), question=str(row.get("question", "")),
                    reference=references,
                    group=str(row.get("question_type", "unknown")))
    # OCRBench's official rule ignores whitespace for the handwritten-maths
    # subset, which is identified by this column.
    if row.get("dataset"):
        sample.meta["subset"] = str(row["dataset"])
    return sample


def adapt_babyvision(row: Row, idx: int) -> Optional[Sample]:
    options = parse_string_list(row.get("options")) or []
    reference = _clean(row.get("answer"))
    if reference is None:
        reference = _clean(row.get("choiceAns") if options else row.get("blankAns"))
    if reference is None:
        reference = _clean(row.get("blankAns")) or _clean(row.get("choiceAns"))
    if reference is None:
        return None

    ans_type = _clean(row.get("ansType")) or ("choice" if options else "blank")
    sample = Sample(uid=str(idx), question=str(row.get("question", "")),
                    options=options or None, reference=reference,
                    group=ans_type)
    sample.meta["ans_type"] = ans_type
    if row.get("type"):
        sample.meta["type"] = str(row["type"])
    return sample


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

register(BenchmarkSpec(
    key="realworldqa",
    task=TaskType.FLEXIBLE,
    description="RealWorldQA spatial reasoning in Kazakh, Russian and English.",
    align=True,
    sources={
        "kk": Source("issai/RealWorldQA_Kazakh_Russian", adapt_realworldqa,
                     config="kazakh", split="test", public=False,
                     image_columns=IMAGE_COLUMNS),
        "ru": Source("issai/RealWorldQA_Kazakh_Russian", adapt_realworldqa,
                     config="russian", split="test", public=False,
                     image_columns=IMAGE_COLUMNS),
        "en": Source("lmms-lab-encoder/RealWorldQA", adapt_realworldqa,
                     config="default", split="test",
                     image_columns=IMAGE_COLUMNS,
                     note="765 items, positionally parallel with the translation"),
    },
))

register(BenchmarkSpec(
    key="mmstar",
    task=TaskType.MCQ,
    description="MMStar multimodal MCQ in Kazakh, Russian and English.",
    align=True,
    max_options=4,
    group_name="category",
    sources={
        "kk": Source("issai/MMstar_Kazakh_Russian", adapt_mmstar,
                     config="kazakh", split="test", public=False,
                     image_columns=IMAGE_COLUMNS),
        "ru": Source("issai/MMstar_Kazakh_Russian", adapt_mmstar,
                     config="russian", split="test", public=False,
                     image_columns=IMAGE_COLUMNS),
        "en": Source("Lin-Chen/MMStar", adapt_mmstar, config="val", split="val",
                     image_columns=IMAGE_COLUMNS,
                     note="1500 items, positionally parallel with the translation"),
    },
))

register(BenchmarkSpec(
    key="ai2d",
    task=TaskType.MCQ,
    description="AI2D science diagram MCQ in Kazakh and English.",
    align=True,
    sources={
        "kk": Source("issai/AI2D_Kazakh", adapt_ai2d, config="default",
                     split="test", public=False, image_columns=IMAGE_COLUMNS),
        "en": Source("lmms-lab-encoder/ai2d", adapt_ai2d, config="default",
                     split="test", image_columns=IMAGE_COLUMNS,
                     note="3088 items, positionally parallel with the translation"),
    },
))

register(BenchmarkSpec(
    key="mathvista",
    task=TaskType.MATH_MIXED,
    description="MathVista visual math (MCQ + open-ended) in Kazakh and English.",
    align=True,
    group_name="mode",
    sources={
        "kk": Source("issai/MathVista_Kazakh", adapt_mathvista, config="default",
                     split="test", public=False, image_columns=IMAGE_COLUMNS),
        "en": Source("AI4Math/MathVista", adapt_mathvista, config="default",
                     split="testmini", image_columns=IMAGE_COLUMNS,
                     note="testmini, 1000 items, positionally parallel"),
    },
))

register(BenchmarkSpec(
    key="mathvision",
    task=TaskType.MATH_MIXED,
    description="MathVision competition visual math in Kazakh and English.",
    align=True,
    group_name="mode",
    sources={
        "kk": Source("issai/MathVision_Kazakh", adapt_mathvision, config="default",
                     split="test", public=False, image_columns=IMAGE_COLUMNS),
        "en": Source("MathLLMs/MathVision", adapt_mathvision, config="default",
                     split="test", image_columns=IMAGE_COLUMNS,
                     note="joined on the MathVision item id"),
    },
))

register(BenchmarkSpec(
    key="mmbench",
    task=TaskType.MCQ,
    description="MMBench multimodal MCQ in Kazakh and English.",
    align=False,
    max_options=4,
    group_name="category",
    sources={
        "kk": Source("issai/MMBench_Kazakh", adapt_mmbench, config="default",
                     split="test", public=False, image_columns=IMAGE_COLUMNS),
        "en": Source("lmms-lab/MMBench_EN", adapt_mmbench, config="default",
                     split="dev", image_columns=IMAGE_COLUMNS,
                     note="NOT parallel with the Kazakh set: the translation was "
                          "re-indexed from zero, so the two item sets differ "
                          "(4329 vs 4377). Compare aggregates, not items."),
    },
))

register(BenchmarkSpec(
    key="ocrbench",
    task=TaskType.OCR_MATCH,
    description="OCRBench text recognition — official rule in English, judged in Kazakh.",
    align=False,
    group_name="question_type",
    sources={
        # The Kazakh subset has a single free-form gold string per item and is
        # graded by the batch judge; the English original ships a list of
        # acceptable strings and uses OCRBench's own containment rule.
        "kk": Source("issai/OCRBench-Kazakh", adapt_ocrbench, config="default",
                     split="test", public=False, image_columns=IMAGE_COLUMNS,
                     task=TaskType.OCR_JUDGE,
                     note="441-item Kazakh subset, not parallel with the English 1000"),
        "en": Source("echo840/OCRBench", adapt_ocrbench, config="default",
                     split="test", image_columns=IMAGE_COLUMNS,
                     note="official OCRBench containment metric, 1000 items"),
    },
))

register(BenchmarkSpec(
    key="babyvision",
    task=TaskType.BABY_MIXED,
    description="BabyVision perception puzzles — choice items exact, blank items judged.",
    align=True,
    group_name="ans_type",
    sources={
        "kk": Source("issai/BabyVision_Kazakh", adapt_babyvision, config="default",
                     split="test", public=False, image_columns=IMAGE_COLUMNS),
        "en": Source("UnipatAI/BabyVision", adapt_babyvision, config="default",
                     split="train", image_columns=IMAGE_COLUMNS,
                     note="388 items, positionally parallel with the translation"),
    },
))
