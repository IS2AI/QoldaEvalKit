"""Row-level parsing helpers shared by dataset adapters.

HF datasets are inconsistent about whether list/dict columns arrive as real
Python objects or as their string repr, so every accessor here is defensive.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from typing import Any, Dict, List, Optional

LETTERS = [chr(ord("A") + i) for i in range(26)]


def parse_string_list(raw: Any) -> Optional[List[str]]:
    """Coerce a column that should be a list of strings."""
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        raw = raw.strip()
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(raw)
            except (json.JSONDecodeError, ValueError, SyntaxError):
                continue
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
    return None


def parse_dict(raw: Any) -> Optional[Dict[str, Any]]:
    """Coerce a column that should be a dict (e.g. ARC ``choices``)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(raw.strip())
            except (json.JSONDecodeError, ValueError, SyntaxError):
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def join_documents(raw: Any) -> str:
    """Flatten a documents column into a single prompt-ready block."""
    docs = parse_string_list(raw)
    if docs is None:
        return str(raw)
    return "\n\n".join(docs)


def to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def gsm8k_gold(answer_text: Any) -> Optional[float]:
    """GSM8K gold answers live after the ``####`` marker."""
    text = str(answer_text)
    match = re.search(r"####\s*(-?[\d,\.]+)", text)
    if match:
        return to_float(match.group(1))
    return to_float(text)


def answer_key_to_index(key: Any, n_options: int, base: int = 0) -> Optional[int]:
    """Normalise an answer key into a 0-based option index.

    ``base`` MUST match the source: 1 for a 1-indexed column, 0 for a 0-indexed
    one. It is not optional — key ``1`` is option A under one convention and
    option B under the other, and guessing shifts every gold answer by one.
    """
    if base not in (0, 1):
        raise ValueError(f"base must be 0 or 1, got {base!r}")

    text = str(key).strip().upper()
    if text in LETTERS[:n_options]:
        return LETTERS.index(text)
    try:
        value = int(float(text))
    except (TypeError, ValueError):
        return None

    index = value - base
    return index if 0 <= index < n_options else None


def stable_shuffle(items: List[Any], key: str) -> List[Any]:
    """Deterministically permute ``items`` using a hash of ``key``.

    Places GPQA's correct answer among its distractors reproducibly.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    ordered = list(items)
    n = len(ordered)
    # Fisher-Yates driven by successive digest bytes.
    for i in range(n - 1, 0, -1):
        j = digest[(n - i) % len(digest)] % (i + 1)
        ordered[i], ordered[j] = ordered[j], ordered[i]
    return ordered
