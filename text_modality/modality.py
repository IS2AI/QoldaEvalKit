"""The text modality: what the shared runner needs to drive text benchmarks."""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.extraction import (
    Completion,
    extract_boxed,
    extract_freeform,
    extract_mcq,
    extract_numeric,
)
from core import schemas
from core.registry import BenchmarkSpec, Sample, TaskType
from core.runner import Modality
from core.scoring import symbolic
from core.scoring.judge import JudgeItem

from . import prompts
from .specs import REGISTRY

NUMERIC_TOLERANCE = 1e-4

METRIC_NAMES = {
    TaskType.SPELLING: "exact_match",
}


class TextModality(Modality):
    name = "text"
    registry = REGISTRY

    def metric_name(self, task: TaskType) -> str:
        return METRIC_NAMES.get(task, "accuracy")

    def response_schema(self, spec: BenchmarkSpec, task: TaskType,
                        sample: Sample) -> Optional[Dict[str, Any]]:
        if task == TaskType.MCQ:
            return schemas.mcq(len(sample.options or []) or spec.max_options or 26)
        if task == TaskType.MATH_NUMERIC:
            return schemas.reasoned_text()      # room to work before answering
        if task == TaskType.MATH_SYMBOLIC:
            return schemas.reasoned_text()
        if task in (TaskType.SPELLING, TaskType.QA_JUDGE, TaskType.RAG_JUDGE):
            return schemas.text()
        # IF_JUDGE: the reply's form is the thing being graded — never constrain.
        return None

    def build_prompt(self, spec: BenchmarkSpec, task: TaskType, sample: Sample,
                     lang: str, prompt_lang: str) -> str:
        return prompts.build_prompt(task, sample, lang, prompt_lang)

    def score(self, spec: BenchmarkSpec, task: TaskType, sample: Sample,
              completion: Completion) -> Dict[str, Any]:
        """Extract the prediction and decide correctness where we can.

        ``correct=None`` means "not decidable here" — either a judge has to
        rule on it, or the symbolic checker gave up.
        """
        answer = completion.answer

        if task == TaskType.MCQ:
            n_options = len(sample.options or []) or (spec.max_options or 26)
            prediction = extract_mcq(completion.for_extraction, n_options)
            return {"prediction": prediction,
                    "correct": bool(prediction and prediction == sample.reference)}

        if task == TaskType.MATH_NUMERIC:
            prediction = extract_numeric(completion.for_extraction)
            correct = (prediction is not None
                       and abs(prediction - float(sample.reference)) < NUMERIC_TOLERANCE)
            return {"prediction": prediction, "correct": correct}

        if task == TaskType.MATH_SYMBOLIC:
            text = completion.for_extraction
            raw = extract_boxed(text) or extract_freeform(text)
            verdict = symbolic.verify(raw, sample.reference)
            return {"prediction": verdict.prediction or raw,
                    "correct": verdict.correct,
                    "score_method": verdict.method}

        if task == TaskType.SPELLING:
            prediction = (extract_freeform(answer) or "").strip()
            return {"prediction": prediction,
                    "correct": prediction == str(sample.reference).strip()}

        if task.is_judged:
            prediction = extract_freeform(answer) or ""
            if not prediction.strip():
                # An empty answer is a model failure, not a grader failure, so
                # it is scored 0 rather than left for the judge to exclude.
                return {"prediction": "", "correct": False,
                        "score_method": "empty_response"}
            # Correctness is settled later, in the batch judge pass.
            return {"prediction": prediction, "correct": None}

        raise ValueError(f"No scorer for task {task}")

    def judge_item(self, spec: BenchmarkSpec, task: TaskType, sample: Sample,
                   record: Dict[str, Any]) -> Optional[JudgeItem]:
        response = str(record.get("prediction") or "").strip()
        if not response:
            return None
        return JudgeItem(
            uid=record["uid"],
            response=response,
            reference=(None if task == TaskType.IF_JUDGE
                       else str(sample.reference)),
            instruction=sample.question,
            constraints=sample.constraints,
        )
