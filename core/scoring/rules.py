"""Rule-based scorers: flexible answer matching and the official OCRBench rule."""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence, Tuple

from ..parsing import LETTERS

_LEAD_IN = re.compile(
    r"^(?:the\s+answer\s+is|answer\s+is|answer|жауап|дұрыс\s+жауап|ответ)\s*[:\-]?\s*",
    re.IGNORECASE,
)


def normalize_answer(text: Any) -> str:
    """Lowercase, strip lead-ins, punctuation and surrounding quotes."""
    value = str(text or "").strip()
    value = value.strip('"').strip("'").strip()
    value = _LEAD_IN.sub("", value).strip()
    value = value.rstrip(".!;,").strip()
    return value.lower()


# Only a comma sitting between digits in groups of three is a thousands
# separator; a comma anywhere else (e.g. the coordinate answer "(4,7)") is
# structure, and must not be dissolved into a number.
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def _as_float(text: str) -> Optional[float]:
    """Parse a string that is *already* a number, modulo currency and percent.

    Deliberately strict: stripping stray characters until something parses
    would make "row 4 col 7" and "(4,7)" both look like 47.
    """
    cleaned = _THOUSANDS.sub("", text)
    cleaned = cleaned.replace("$", "").replace("%", "").replace(" ", "").strip()
    cleaned = cleaned.rstrip(".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def flexible_match(prediction: Optional[str], reference: Any,
                   precision: Optional[int] = None) -> bool:
    """Match an answer that may be a letter, a word or a number.

    ``precision`` rounds both sides before comparing, which is what MathVista's
    ``precision`` column asks for on float answers.
    """
    if prediction is None:
        return False
    pred, ref = normalize_answer(prediction), normalize_answer(reference)
    if not ref:
        return False
    if pred == ref:
        return True
    # A bare letter answer against a bare letter reference.
    if len(pred) == 1 and len(ref) == 1 and pred.upper() == ref.upper():
        return True

    pred_number, ref_number = _as_float(pred), _as_float(ref)
    if pred_number is not None and ref_number is not None:
        if precision is not None:
            return round(pred_number, precision) == round(ref_number, precision)
        return abs(pred_number - ref_number) < 1e-6
    return False


def find_option_letter(choices: Sequence[Any], answer: Any) -> Optional[str]:
    """Which option letter holds ``answer``.

    MathVista and MathVision give the correct *value* rather than an index, so
    the letter has to be recovered by matching against the option list.
    """
    target = normalize_answer(answer)
    for index, choice in enumerate(choices):
        if index >= len(LETTERS):
            break
        if normalize_answer(choice) == target:
            return LETTERS[index]
        choice_number, target_number = _as_float(normalize_answer(choice)), _as_float(target)
        if (choice_number is not None and target_number is not None
                and abs(choice_number - target_number) < 1e-6):
            return LETTERS[index]
    return None


# ---------------------------------------------------------------------------
# ASR: word and character error rate
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s']", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize_transcript(text: Any) -> str:
    """Lowercase, drop punctuation, collapse whitespace.

    This is the usual ASR normalisation; the un-normalised rate is reported
    alongside it so the effect is always visible.
    """
    value = str(text or "").lower()
    value = _PUNCT.sub(" ", value)
    return _SPACES.sub(" ", value).strip()


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Levenshtein distance over token (or character) sequences."""
    if not reference:
        return len(hypothesis)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            cost = 0 if ref_token == hyp_token else 1
            current.append(min(previous[j] + 1,        # deletion
                               current[j - 1] + 1,     # insertion
                               previous[j - 1] + cost))  # substitution
        previous = current
    return previous[-1]


def error_rate(reference: Any, hypothesis: Any, unit: str = "word",
               normalize: bool = True) -> Tuple[int, int]:
    """Return (edits, reference_length) so rates can be pooled corpus-wide.

    Corpus WER is ``sum(edits) / sum(lengths)`` — not the mean of per-utterance
    rates, which over-weights short clips.
    """
    ref = normalize_transcript(reference) if normalize else str(reference or "").strip()
    hyp = normalize_transcript(hypothesis) if normalize else str(hypothesis or "").strip()
    ref_tokens = list(ref) if unit == "char" else ref.split()
    hyp_tokens = list(hyp) if unit == "char" else hyp.split()
    return edit_distance(ref_tokens, hyp_tokens), len(ref_tokens)


# ---------------------------------------------------------------------------
# OCRBench
# ---------------------------------------------------------------------------

# The official script compares whitespace-stripped strings for the handwritten
# maths subset and lowercased substrings everywhere else.
_OCR_NO_SPACE_SUBSETS = {"hme100k"}


def ocr_match(prediction: Optional[str], references: Any,
              subset: Optional[str] = None) -> bool:
    """The official OCRBench rule: the gold string appears in the response.

    ``references`` may be a single string or a list of acceptable answers; any
    one of them matching counts as correct.
    """
    if not prediction:
        return False

    if isinstance(references, (list, tuple)):
        candidates = [str(r) for r in references]
    else:
        candidates = [str(references)]

    strip_spaces = str(subset or "").strip().lower() in _OCR_NO_SPACE_SUBSETS

    for reference in candidates:
        gold = reference.strip().replace("\n", " ")
        guess = prediction.strip().replace("\n", " ")
        if strip_spaces:
            gold = gold.replace(" ", "")
            guess = guess.replace(" ", "")
        else:
            gold, guess = gold.lower(), guess.lower()
        if gold and gold in guess:
            return True
    return False
