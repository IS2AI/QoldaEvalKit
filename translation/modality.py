"""The translation modality: translate FLORES, score with XCOMET."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from core.extraction import Completion, find_json_objects
from core import schemas
from core.registry import BenchmarkSpec, Sample, TaskType
from core.runner import Modality

from .metrics import XCometScorer, surface_metrics
from .specs import LANGUAGE_NAMES, REGISTRY

# "Translation:" / "Аударма:" / "Перевод:" and friends, when a model insists on
# labelling its output despite being asked not to.
_LEAD_IN = re.compile(
    r"^\s*(?:translation|translated text|аударма|перевод)\s*[:\-]\s*",
    re.IGNORECASE,
)


# Opening quote -> its closing partner. Symmetric ones map to themselves.
_QUOTE_PAIRS = {'"': '"', "'": "'", "\u00ab": "\u00bb",
                "\u201c": "\u201d", "\u201e": "\u201c", "\u2018": "\u2019"}


def extract_translation(answer: str) -> str:
    """The translated text alone. Conservative: a translation is the whole
    turn, so nothing is dropped unless it is clearly a wrapper."""
    text = (answer or "").strip()
    if not text:
        return ""

    for span in reversed(find_json_objects(text)):
        try:
            obj = json.loads(span)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            for key in ("translation", "answer", "text", "output"):
                if obj.get(key):
                    return str(obj[key]).strip()

    text = _LEAD_IN.sub("", text).strip()
    # A fully-quoted line is a wrapper; quotes inside a sentence are not.
    if len(text) > 1 and _QUOTE_PAIRS.get(text[0]) == text[-1]:
        inner = text[1:-1].strip()
        if inner:
            text = inner
    return text.strip()


class TranslationModality(Modality):
    name = "translation"
    registry = REGISTRY

    def __init__(self):
        # One scorer for the whole run; the checkpoint is large.
        self._scorer: Optional[XCometScorer] = None
        self._config = None

    def metric_name(self, task: TaskType) -> str:
        return "xcomet"

    def build_prompt(self, spec: BenchmarkSpec, task: TaskType, sample: Sample,
                     lang: str, prompt_lang: str) -> str:
        source = LANGUAGE_NAMES[sample.meta["source_lang"]]
        target = LANGUAGE_NAMES[sample.meta["target_lang"]]
        return (f"Translate the following {source} text into {target}.\n"
                f"Return only the translation, with no explanation, no notes "
                f"and no quotation marks.\n\n"
                f"{source} text: {sample.question}")

    def response_schema(self, spec: BenchmarkSpec, task: TaskType,
                        sample: Sample) -> Optional[Dict[str, Any]]:
        return schemas.text("translation", "translation")

    def score(self, spec: BenchmarkSpec, task: TaskType, sample: Sample,
              completion: Completion) -> Dict[str, Any]:
        translation = extract_translation(completion.answer)
        # Not a per-item boolean; XCOMET scores the whole set in post_score.
        return {"prediction": translation, "correct": None,
                "score_method": "xcomet"}

    # -- whole-set scoring -------------------------------------------------

    def _get_scorer(self) -> XCometScorer:
        if self._scorer is None:
            cfg = getattr(self._config, "translation", None)
            self._scorer = XCometScorer(
                model_name=getattr(cfg, "model", "Unbabel/XCOMET-XXL"),
                batch_size=getattr(cfg, "batch_size", 8),
                gpus=getattr(cfg, "gpus", 1),
                cache_dir=getattr(cfg, "cache_dir", None) or None,
            )
        return self._scorer

    async def post_score(self, spec: BenchmarkSpec, task: TaskType, lang: str,
                         samples: List[Sample],
                         records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        import asyncio

        cfg = getattr(self._config, "translation", None)
        if cfg is not None and not getattr(cfg, "enabled", True):
            print("   xcomet disabled — translations saved, not scored")
            return records

        by_uid = {s.uid: s for s in samples}
        gradable, triplets = [], []
        for record in records:
            hypothesis = str(record.get("prediction") or "").strip()
            sample = by_uid.get(record["uid"])
            if not hypothesis or sample is None:
                # An empty translation scores zero rather than vanishing from
                # the denominator: producing nothing is a translation failure.
                record["xcomet"] = 0.0
                record["correct"] = False
                continue
            gradable.append(record)
            triplets.append({"src": sample.question,
                             "mt": hypothesis,
                             "ref": str(sample.reference)})

        if not triplets:
            return records

        scorer = self._get_scorer()
        try:
            scores, system = await asyncio.to_thread(scorer.score, triplets)
        except Exception as exc:  # noqa: BLE001
            # Leave translations on disk so --resume can rescore them.
            print(f"   xcomet failed: {exc}")
            for record in gradable:
                record["xcomet"] = None
                record["correct"] = None
            return records

        for record, score in zip(gradable, scores):
            record["xcomet"] = round(float(score), 6)
            record["correct"] = True   # scored; the value lives in `xcomet`
        if system is not None:
            print(f"   xcomet system score: {system * 100:.2f}")
        return records

    # -- aggregation -------------------------------------------------------

    def aggregate(self, spec: BenchmarkSpec, task: TaskType,
                  records: List[Dict[str, Any]]) -> Dict[str, Any]:
        scored = [r for r in records if isinstance(r.get("xcomet"), (int, float))]
        empty = sum(1 for r in records
                    if not str(r.get("prediction") or "").strip())
        translated = len(records) - empty

        block: Dict[str, Any] = {
            "xcomet": (round(sum(r["xcomet"] for r in scored) / len(scored), 6)
                       if scored else None),
            "total_samples": len(records),
            "extraction_failed": empty,
            "truncated_thinking_samples": sum(1 for r in records
                                              if r.get("truncated_thinking")),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        target_lang = spec.key.rsplit("_", 1)[-1]
        block.update(surface_metrics(
            [str(r.get("prediction") or "") for r in records],
            [str(r.get("reference") or "") for r in records],
            target_lang,
        ))

        # XCOMET leads when it ran, BLEU when it did not. Both are reported
        # as plain fractions rather than percentages.
        block["display"] = "decimal"

        if scored:
            block["metric"] = "xcomet"
            block["scored_samples"] = len(scored)
            block["unscorable_samples"] = len(records) - len(scored)
        elif isinstance(block.get("bleu"), (int, float)):
            block["metric"] = "bleu"
            block["scored_samples"] = translated
            block["unscorable_samples"] = empty
        else:
            # No XCOMET and no sacrebleu: translations are saved, nothing scored.
            block["metric"] = "xcomet"
            block["scored_samples"] = 0
            block["unscorable_samples"] = len(records)
            block["note"] = ("not scored — xcomet disabled and sacrebleu is not "
                             "installed (pip install sacrebleu)")

        if spec.group_name:
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for record in scored:
                groups.setdefault(str(record.get("group", "unknown")), []).append(record)
            block[f"by_{spec.group_name}"] = {
                name: {"xcomet": round(sum(r["xcomet"] for r in items) / len(items), 6),
                       "n": len(items)}
                for name, items in sorted(groups.items())
            }
        return block
