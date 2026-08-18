"""Qorgau safety benchmark definition.

Qorgau (arXiv:2502.13640) is a Kazakh–Russian safety set: adversarial questions
across six risk areas, asked in the model's own language and graded against a
per-risk-area rubric.  The workbook ships one sheet per language and there is
no English side, so only kk and ru are registered.

The dataset file lives in ``datasets/ru_kz_question_only.xlsx``, copied from the
Qorgau repository.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.registry import BenchmarkSpec, Registry, Sample, Source, TaskType

from .rubrics import RISK_AREA_SHORT, RISK_AREA_TO_ID

REGISTRY = Registry()
register = REGISTRY.register

Row = Dict[str, Any]

DATASET_FILE = "ru_kz_question_only.xlsx"

# The Kazakh sheet's id column is literally named " ".
_ID_KEYS = ("id", " ", "Unnamed: 0")


def _first(row: Row, keys) -> Optional[str]:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return None


def adapt_qorgau(row: Row, idx: int) -> Optional[Sample]:
    question = row.get("question")
    if question is None or not str(question).strip():
        return None
    risk_area = str(row.get("risk_area") or "").strip()
    risk_id = RISK_AREA_TO_ID.get(risk_area)
    if risk_id is None:
        return None

    item_id = _first(row, _ID_KEYS) or str(idx)
    question_type = str(row.get("question_type") or "original").strip()
    # An id repeats across question types (original / task1-FN / task2-FP), so
    # both are needed for a unique key.
    sample = Sample(uid=f"{item_id}:{question_type}",
                    question=str(question).strip(),
                    reference=None,
                    group=RISK_AREA_SHORT[risk_id])
    sample.meta.update({
        "risk_id": risk_id,
        "risk_area": risk_area,
        "risk_area_specified": row.get("risk_area_specified"),
        "types_of_harm": row.get("types_of_harm"),
        "question_type": question_type,
    })
    return sample


register(BenchmarkSpec(
    key="qorgau",
    task=TaskType.SAFETY_RUBRIC,
    description="Qorgau — Kazakh/Russian LLM safety across six risk areas.",
    align=False,     # the two sheets are not parallel (3786 kk vs 4383 ru)
    group_name="risk_area",
    sources={
        "kk": Source(DATASET_FILE, adapt_qorgau, kind="xlsx", config="Kazakh",
                     note="Qorgau Kazakh sheet; graded with the Kazakh rubric"),
        "ru": Source(DATASET_FILE, adapt_qorgau, kind="xlsx", config="Russian",
                     note="Qorgau Russian sheet; graded with the Russian rubric"),
    },
))
