"""The vision modality: prompt building and scoring for image benchmarks."""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.extraction import (
    Completion,
    extract_boxed,
    extract_flexible,
    extract_freeform,
    extract_mcq,
)
from core.parsing import LETTERS
from core import schemas
from core.registry import BenchmarkSpec, Sample, TaskType
from core.runner import Modality
from core.scoring import symbolic
from core.scoring.judge import JudgeItem
from core.scoring.rules import find_option_letter, flexible_match, ocr_match

from . import prompts
from .specs import REGISTRY


def _gold_letter(sample: Sample) -> Optional[str]:
    """The expected option letter, whether the dataset stores a letter or a value."""
    options = sample.options or []
    reference = str(sample.reference or "").strip()
    if reference.upper() in LETTERS[:len(options)]:
        return reference.upper()
    return find_option_letter(options, reference)


class VisionModality(Modality):
    name = "vision"
    registry = REGISTRY

    def metric_name(self, task: TaskType) -> str:
        return "accuracy"

    def response_schema(self, spec: BenchmarkSpec, task: TaskType,
                        sample: Sample) -> Optional[Dict[str, Any]]:
        if task == TaskType.MCQ:
            return schemas.mcq(len(sample.options or []) or spec.max_options or 26)
        if task == TaskType.MATH_MIXED:
            return (schemas.mcq(len(sample.options)) if sample.options
                    else schemas.reasoned_text())
        if task in (TaskType.OCR_MATCH, TaskType.OCR_JUDGE, TaskType.FLEXIBLE):
            return schemas.text()
        if task == TaskType.BABY_MIXED:
            return (schemas.mcq(len(sample.options)) if sample.options
                    else schemas.text())
        return None

    def build_prompt(self, spec: BenchmarkSpec, task: TaskType, sample: Sample,
                     lang: str, prompt_lang: str) -> str:
        return prompts.build_prompt(task, sample, lang, prompt_lang)

    # -- scoring -----------------------------------------------------------

    def score(self, spec: BenchmarkSpec, task: TaskType, sample: Sample,
              completion: Completion) -> Dict[str, Any]:
        answer = completion.answer

        if task == TaskType.MCQ:
            n_options = len(sample.options or []) or (spec.max_options or 26)
            prediction = extract_mcq(completion.for_extraction, n_options)
            return {"prediction": prediction,
                    "correct": bool(prediction
                                    and prediction == str(sample.reference).upper()),
                    "score_method": "letter"}

        if task == TaskType.FLEXIBLE:
            prediction = extract_flexible(answer)
            return {"prediction": prediction,
                    "correct": flexible_match(prediction, sample.reference),
                    "score_method": "flexible"}

        if task == TaskType.MATH_MIXED:
            return self._score_math_mixed(spec, sample, completion.for_extraction)

        if task == TaskType.OCR_MATCH:
            prediction = extract_freeform(answer) or ""
            return {"prediction": prediction,
                    "correct": ocr_match(prediction, sample.reference,
                                         sample.meta.get("subset")),
                    "score_method": "ocrbench_rule"}

        if task == TaskType.OCR_JUDGE:
            prediction = extract_freeform(answer) or ""
            if not prediction.strip():
                return {"prediction": "", "correct": False,
                        "score_method": "empty_response"}
            return {"prediction": prediction, "correct": None}

        if task == TaskType.BABY_MIXED:
            return self._score_baby(sample, completion.for_extraction)

        raise ValueError(f"No vision scorer for task {task}")

    def _score_math_mixed(self, spec: BenchmarkSpec, sample: Sample,
                          answer: str) -> Dict[str, Any]:
        """MathVista / MathVision: letters when the item has options, symbolic
        verification when it is open-ended."""
        if sample.options:
            gold = _gold_letter(sample)
            if gold is not None:
                n_options = len(sample.options)
                prediction = extract_mcq(answer, n_options)
                return {"prediction": prediction,
                        "correct": bool(prediction and prediction == gold),
                        "score_method": "letter"}
            # The gold value is not one of the listed options; fall through and
            # compare the free-form answer instead of guessing a letter.

        prediction = extract_boxed(answer) or extract_flexible(answer)
        precision = sample.meta.get("precision")
        if flexible_match(prediction, sample.reference, precision):
            return {"prediction": prediction, "correct": True,
                    "score_method": "flexible"}

        verdict = symbolic.verify(prediction, sample.reference)
        return {"prediction": verdict.prediction or prediction,
                "correct": verdict.correct,
                "score_method": verdict.method}

    def _score_baby(self, sample: Sample, answer: str) -> Dict[str, Any]:
        """Choice items are settled exactly; only free-form items reach the judge."""
        if sample.options:
            gold = _gold_letter(sample)
            if gold is not None:
                prediction = extract_mcq(answer, len(sample.options))
                return {"prediction": prediction,
                        "correct": bool(prediction and prediction == gold),
                        "score_method": "letter"}

        prediction = extract_flexible(answer) or ""
        if not prediction.strip():
            return {"prediction": "", "correct": False,
                    "score_method": "empty_response"}
        if flexible_match(prediction, sample.reference):
            # An exact hit needs no judge.
            return {"prediction": prediction, "correct": True,
                    "score_method": "flexible"}
        return {"prediction": prediction, "correct": None}

    # -- judging -----------------------------------------------------------

    def judge_item(self, spec: BenchmarkSpec, task: TaskType, sample: Sample,
                   record: Dict[str, Any]) -> Optional[JudgeItem]:
        response = str(record.get("prediction") or "").strip()
        if not response:
            return None
        reference = sample.reference
        if isinstance(reference, (list, tuple)):
            reference = " | ".join(str(r) for r in reference)
        return JudgeItem(uid=record["uid"], response=response,
                         reference=str(reference))
