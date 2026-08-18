"""FLORES translation benchmark definitions.

One benchmark per direction, so a direction can be selected on its own and the
summary reports each separately:

    flores_kk_en  flores_kk_ru  flores_ru_kk  flores_ru_en
    flores_en_kk  flores_en_ru

Each is registered under its *source* language, so ``LANGUAGES=kk`` runs the two
directions out of Kazakh, and ``LANGUAGES=all`` runs all six.

The data is ``datasets/flores_v2.csv``, exported from OylanEvalTeam/flores-v2.0
with the Turkish columns removed.  Chinese columns are kept in the file but no
direction uses them, since the toolkit's language set is kk/ru/en.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Optional

from core.registry import BenchmarkSpec, Registry, Sample, Source, TaskType

REGISTRY = Registry()
register = REGISTRY.register

Row = Dict[str, Any]

DATASET_FILE = "flores_v2.csv"

# Toolkit language code -> the column prefix used in the FLORES export.
COLUMN_PREFIX = {"kk": "kaz", "ru": "rus", "en": "eng"}

LANGUAGE_NAMES = {"kk": "Kazakh", "ru": "Russian", "en": "English"}

# FLORES ships dev + devtest; devtest is the evaluation half.
EVAL_SPLIT = "devtest"

DIRECTIONS = [
    ("kk", "en"), ("kk", "ru"),
    ("ru", "kk"), ("ru", "en"),
    ("en", "kk"), ("en", "ru"),
]


def adapt_flores(row: Row, idx: int, source_lang: str,
                 target_lang: str) -> Optional[Sample]:
    """One parallel sentence pair, as a translation item."""
    src_prefix = COLUMN_PREFIX[source_lang]
    tgt_prefix = COLUMN_PREFIX[target_lang]

    split = row.get(f"{src_prefix}_split")
    if split and split != EVAL_SPLIT:
        return None

    source_text = row.get(f"{src_prefix}_text")
    target_text = row.get(f"{tgt_prefix}_text")
    if not source_text or not target_text:
        return None

    sample = Sample(uid=str(row.get("id", idx)),
                    question=str(source_text).strip(),
                    reference=str(target_text).strip(),
                    group=str(row.get(f"{src_prefix}_domain") or "unknown"))
    sample.meta.update({
        "source_lang": source_lang,
        "target_lang": target_lang,
        "topic": row.get(f"{src_prefix}_topic"),
    })
    return sample


for _src, _tgt in DIRECTIONS:
    register(BenchmarkSpec(
        key=f"flores_{_src}_{_tgt}",
        task=TaskType.TRANSLATION,
        description=(f"FLORES {LANGUAGE_NAMES[_src]} → {LANGUAGE_NAMES[_tgt]}"
                     f" ({EVAL_SPLIT}), scored with XCOMET."),
        group_name="domain",
        sources={
            _src: Source(DATASET_FILE,
                         partial(adapt_flores, source_lang=_src, target_lang=_tgt),
                         kind="csv",
                         note=f"FLORES v2.0 {EVAL_SPLIT}, {_src}->{_tgt}"),
        },
    ))
