"""The audio modality: prompt building, scoring and WER aggregation."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from core.extraction import (
    Completion,
    extract_flexible,
    extract_mcq,
    extract_numeric,
)
from core import schemas
from core.registry import BenchmarkSpec, Sample, TaskType
from core.runner import Modality
from core.scoring.judge import JudgeItem
from core.scoring.rules import error_rate, normalize_transcript

from . import prompts
from .specs import REGISTRY

NUMERIC_TOLERANCE = 1e-4

METRIC_NAMES = {
    TaskType.ASR_WER: "wer",
}


class AudioModality(Modality):
    name = "audio"
    registry = REGISTRY

    def metric_name(self, task: TaskType) -> str:
        return METRIC_NAMES.get(task, "accuracy")

    def response_schema(self, spec: BenchmarkSpec, task: TaskType,
                        sample: Sample) -> Optional[Dict[str, Any]]:
        if task == TaskType.MCQ:
            return schemas.mcq(len(sample.options or []) or spec.max_options or 4)
        if task == TaskType.MATH_NUMERIC:
            return schemas.reasoned_text()
        if task in (TaskType.ASR_WER, TaskType.AUDIO_JUDGE):
            return schemas.text()
        return None

    def build_prompt(self, spec: BenchmarkSpec, task: TaskType, sample: Sample,
                     lang: str, prompt_lang: str) -> str:
        return prompts.build_prompt(task, sample, lang, prompt_lang)

    # -- scoring -----------------------------------------------------------

    def score(self, spec: BenchmarkSpec, task: TaskType, sample: Sample,
              completion: Completion) -> Dict[str, Any]:
        answer = completion.answer

        if task == TaskType.MCQ:
            n_options = len(sample.options or []) or (spec.max_options or 4)
            prediction = extract_mcq(completion.for_extraction, n_options)
            return {"prediction": prediction,
                    "correct": bool(prediction
                                    and prediction == str(sample.reference).upper()),
                    "score_method": "letter"}

        if task == TaskType.MATH_NUMERIC:
            prediction = extract_numeric(completion.for_extraction)
            correct = (prediction is not None
                       and abs(prediction - float(sample.reference)) < NUMERIC_TOLERANCE)
            return {"prediction": prediction, "correct": correct,
                    "score_method": "numeric"}

        if task == TaskType.ASR_WER:
            hypothesis = (extract_flexible(answer) or "").strip()
            reference = str(sample.reference)
            edits, ref_len = error_rate(reference, hypothesis, unit="word")
            raw_edits, raw_len = error_rate(reference, hypothesis, unit="word",
                                            normalize=False)
            char_edits, char_len = error_rate(reference, hypothesis, unit="char")
            return {
                "prediction": hypothesis,
                # WER is pooled corpus-wide, so per-item correctness is not a
                # meaningful concept here; the counts below carry the score.
                "correct": None,
                "score_method": "wer",
                "wer": round(edits / ref_len, 6) if ref_len else None,
                "edits": edits, "ref_len": ref_len,
                "raw_edits": raw_edits, "raw_ref_len": raw_len,
                "char_edits": char_edits, "char_ref_len": char_len,
            }

        if task == TaskType.AUDIO_JUDGE:
            prediction = extract_flexible(answer) or ""
            if not prediction.strip():
                return {"prediction": "", "correct": False,
                        "score_method": "empty_response"}
            return {"prediction": prediction, "correct": None}

        raise ValueError(f"No audio scorer for task {task}")

    # -- judging -----------------------------------------------------------

    def judge_item(self, spec: BenchmarkSpec, task: TaskType, sample: Sample,
                   record: Dict[str, Any]) -> Optional[JudgeItem]:
        if task != TaskType.AUDIO_JUDGE:
            return None
        response = str(record.get("prediction") or "").strip()
        if not response:
            return None
        return JudgeItem(uid=record["uid"], response=response,
                         reference=str(sample.reference))

    # -- aggregation -------------------------------------------------------

    def aggregate(self, spec: BenchmarkSpec, task: TaskType,
                  records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if task != TaskType.ASR_WER:
            return super().aggregate(spec, task, records)
        return _aggregate_wer(records, spec.group_name)


def _corpus_rate(records: List[Dict[str, Any]], edits_key: str,
                 length_key: str) -> Optional[float]:
    """Pool edits over the whole set rather than averaging per-utterance rates,
    which would over-weight short clips."""
    total_edits = sum(r.get(edits_key) or 0 for r in records)
    total_length = sum(r.get(length_key) or 0 for r in records)
    return round(total_edits / total_length, 6) if total_length else None


def _aggregate_wer(records: List[Dict[str, Any]],
                   group_name: Optional[str]) -> Dict[str, Any]:
    total = len(records)
    scored = [r for r in records if r.get("ref_len")]
    empty = sum(1 for r in records if not str(r.get("prediction") or "").strip())

    block: Dict[str, Any] = {
        "metric": "wer",
        "lower_is_better": True,
        # WER is a rate, read as 0.1707 rather than 17.07%.
        "display": "decimal",
        "wer": _corpus_rate(scored, "edits", "ref_len"),
        "wer_raw": _corpus_rate(scored, "raw_edits", "raw_ref_len"),
        "cer": _corpus_rate(scored, "char_edits", "char_ref_len"),
        "total_samples": total,
        "scored_samples": len(scored),
        "extraction_failed": empty,
        "truncated_thinking_samples": sum(1 for r in records
                                          if r.get("truncated_thinking")),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if group_name:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for record in scored:
            groups.setdefault(str(record.get("group", "unknown")), []).append(record)
        block[f"by_{group_name}"] = {
            name: {"wer": _corpus_rate(items, "edits", "ref_len"),
                   "n": len(items)}
            for name, items in sorted(groups.items())
        }
    return block
