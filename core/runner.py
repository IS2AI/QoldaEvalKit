"""Generic run orchestration: load -> generate -> score -> judge -> report.

Everything modality-specific (prompt wording, extraction, scoring rules) is
supplied by a :class:`Modality` object, so text and vision share this loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from .config import RunConfig
from .extraction import Completion
from .generation import ModelClient
from .audio import encode_audio
from .images import encode_image
from .loaders import load_benchmark
from .report_md import write_performance_report
from .registry import BenchmarkSpec, Registry, Sample, TaskType
from .reporting import (
    aggregate as default_aggregate,
    has_prediction,
    load_records,
    print_table,
    records_path,
    write_records,
    write_summary,
)
from .scoring import judge as judge_mod

logger = logging.getLogger("qoldaevalkit")

# Evaluation order: all Kazakh first, then English, then Russian.
LANGUAGE_ORDER = ("kk", "en", "ru")


class Modality:
    """What a modality has to provide for the shared runner to drive it.

    Subclasses must implement ``build_prompt``, ``score`` and ``judge_item``.
    ``metric_name`` and ``aggregate`` have sensible defaults and are only
    overridden where a benchmark reports something other than accuracy — ASR,
    which is scored by word error rate.
    """

    name: str = "modality"
    registry: Registry

    def attach_config(self, config: RunConfig) -> None:
        """Hand the modality the run's settings before evaluation starts."""
        self._config = config

    def build_prompt(self, spec: BenchmarkSpec, task: TaskType, sample: Sample,
                     lang: str, prompt_lang: str) -> str:
        raise NotImplementedError

    def score(self, spec: BenchmarkSpec, task: TaskType, sample: Sample,
              completion: Completion) -> Dict[str, Any]:
        raise NotImplementedError

    def judge_item(self, spec: BenchmarkSpec, task: TaskType, sample: Sample,
                   record: Dict[str, Any]) -> Optional[judge_mod.JudgeItem]:
        return None

    def metric_name(self, task: TaskType) -> str:
        return "accuracy"

    async def post_score(self, spec: BenchmarkSpec, task: TaskType, lang: str,
                         samples: List[Sample],
                         records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score the whole set at once, after generation.

        For metrics that are neither per-item nor an API judge — a local neural
        metric such as XCOMET, which wants every segment in one batch.
        """
        return records

    def response_schema(self, spec: BenchmarkSpec, task: TaskType,
                        sample: Sample) -> Optional[Dict[str, Any]]:
        """A JSON schema for the evaluated model's reply, or None to leave it
        free-form. Only consulted when --structured_output is on."""
        return None

    def judge_raw(self, task: TaskType) -> bool:
        """True when this task wants the judge's reply text, not a 0/1."""
        return False

    def apply_verdict(self, spec: BenchmarkSpec, task: TaskType, sample: Sample,
                      record: Dict[str, Any], verdict: Any) -> None:
        """Fold one judge verdict back into its record."""
        record["correct"] = bool(verdict)
        record["score_method"] = "llm_judge"

    def aggregate(self, spec: BenchmarkSpec, task: TaskType,
                  records: List[Dict[str, Any]]) -> Dict[str, Any]:
        return default_aggregate(records, self.metric_name(task),
                                 spec.group_name)


class Evaluator:
    def __init__(self, config: RunConfig, modality: Modality):
        self.config = config
        self.modality = modality
        self.client = ModelClient(config.model, config.sampling, config.image,
                                  config.audio)
        modality.attach_config(config)
        # One gate for the whole per-item pipeline (see process() below).
        self._gate = asyncio.Semaphore(config.model.concurrency)
        # Benchmarks are loaded once and reused across the language-major loop.
        self._data_cache: Dict[str, Any] = {}

    # -- one benchmark x language -----------------------------------------

    async def _images_for(self, sample: Sample) -> Optional[List[str]]:
        """Decode and encode this item's image off the event loop."""
        if not sample.has_image:
            return None

        def work() -> Optional[str]:
            try:
                return encode_image(sample.image_loader(), self.config.image)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Image decode failed for %s: %s", sample.uid, exc)
                return None

        encoded = await asyncio.to_thread(work)
        return [encoded] if encoded else None

    async def _audio_for(self, sample: Sample) -> Optional[List[str]]:
        """Decode and encode this item's clip off the event loop."""
        if not sample.has_audio:
            return None

        def work() -> Optional[str]:
            try:
                return encode_audio(sample.audio_loader(), self.config.audio)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Audio decode failed for %s: %s", sample.uid, exc)
                return None

        encoded = await asyncio.to_thread(work)
        return [encoded] if encoded else None

    async def _evaluate(self, spec: BenchmarkSpec, lang: str, task: TaskType,
                        samples: List[Sample]) -> List[Dict[str, Any]]:
        path = records_path(self.config.run_dir, spec.key, lang)

        cached: Dict[str, Dict[str, Any]] = {}
        if self.config.resume:
            # Every record on disk counts as generated, unparsed ones
            # included; the retry budget below decides re-attempts.
            cached = {r["uid"]: r for r in load_records(path)}
            if cached:
                blank = sum(1 for r in cached.values() if not has_prediction(r))
                note = f", {blank} with no parseable answer" if blank else ""
                print(f"  resuming: {len(cached)} of {len(samples)} "
                      f"items already done{note}")

        pending = [s for s in samples if s.uid not in cached]

        async def process(sample: Sample) -> Dict[str, Any]:
            # Gates decoding and the request together, so only in-flight
            # items hold an encoded payload in memory.
            async with self._gate:
                prompt = self.modality.build_prompt(
                    spec, task, sample, lang, self.config.prompt_lang
                )
                images = await self._images_for(sample)
                audios = await self._audio_for(sample)
                schema = (self.modality.response_schema(spec, task, sample)
                          if self.config.model.structured_output else None)
                completion = await self.client.generate(prompt, images, audios,
                                                        schema)

            record: Dict[str, Any] = {
                "uid": sample.uid,
                "benchmark": spec.key,
                "language": lang,
                "reference": sample.reference,
                "group": sample.group,
                "finish_reason": completion.finish_reason,
                "truncated_thinking": completion.truncated_thinking,
            }
            if sample.meta:
                record["meta"] = dict(sample.meta)
            try:
                record.update(self.modality.score(spec, task, sample, completion))
            except Exception as exc:  # noqa: BLE001
                # One unscoreable item must not abort the benchmark; it is
                # recorded as ungraded and counted as unscorable.
                logger.warning("Scoring failed for %s/%s %s: %s: %s",
                               spec.key, lang, sample.uid,
                               type(exc).__name__, exc)
                record.update({"prediction": None, "correct": None,
                               "score_method": "scorer_error",
                               "error": f"{type(exc).__name__}: {exc}"[:300]})
            if self.config.save_responses:
                record["thinking"] = completion.reasoning
                record["response"] = completion.answer
            return record

        from tqdm.asyncio import tqdm as tqdm_async

        results: List[Dict[str, Any]] = []
        if pending:
            results = await tqdm_async.gather(
                *[process(s) for s in pending],
                desc=f"{spec.key}/{lang}", leave=False,
            )

        by_uid = dict(cached)
        for record in results:
            by_uid[record["uid"]] = record

        # Second pass over items with no readable answer. Only empty
        # predictions are retried, at unchanged sampling. The budget is per
        # item and persists in the record, so a resume does not re-spend it.
        budget = self.config.retry_unparsed
        spent: Dict[str, int] = {r["uid"]: 0 for r in results}
        for uid, record in cached.items():
            # No counter means a run from before this bookkeeping; treat the
            # item as finished rather than re-spending on it.
            spent[uid] = int(record.get("retry_attempts", budget))

        for attempt in range(1, budget + 1):
            stale = [s for s in samples
                     if s.uid in by_uid
                     and not has_prediction(by_uid[s.uid])
                     and spent.get(s.uid, 0) < budget]
            if not stale:
                break
            print(f"  retry {attempt}/{budget}: "
                  f"{len(stale)} item(s) with no parseable answer")
            retried = await tqdm_async.gather(
                *[process(s) for s in stale],
                desc=f"{spec.key}/{lang} retry{attempt}", leave=False,
            )
            recovered = 0
            for record in retried:
                uid = record["uid"]
                spent[uid] = spent.get(uid, 0) + 1
                # Keep the retry only if it produced an answer; either way
                # record the spend so a later resume can see it.
                if has_prediction(record):
                    record["retry_attempts"] = spent[uid]
                    by_uid[uid] = record
                    recovered += 1
                else:
                    by_uid[uid]["retry_attempts"] = spent[uid]
            print(f"  retry {attempt}: recovered {recovered} of {len(stale)}")
            if not recovered:
                break      # nothing is being gained; further passes won't help

        ordered = [by_uid[s.uid] for s in samples if s.uid in by_uid]
        write_records(path, ordered)
        return ordered

    # -- judging -----------------------------------------------------------

    async def _judge_all(self, pending: List[tuple]) -> None:
        """Grade every deferred benchmark in one pass.

        Jobs from all benchmarks and languages are planned together, so small
        sets go direct while large ones batch, and several batches are in
        flight at once instead of one benchmark waiting on the next.
        """
        jobs = []
        owners: Dict[str, tuple] = {}

        for spec, lang, task, samples, records in pending:
            if self.config.rejudge:
                # Only verdicts the judge produced are cleared; items settled
                # locally have nothing for a judge to read.
                stale = [r for r in records
                         if r.get("score_method") == "llm_judge"]
                for record in stale:
                    record["correct"] = None
                if stale:
                    print(f"  re-judging {len(stale)} settled verdict(s) in "
                          f"{spec.key} [{lang}]")

            owner = f"{spec.key}.{lang}"
            by_uid = {s.uid: s for s in samples}
            items = []
            for record in records:
                if record.get("correct") is not None:
                    continue          # already settled locally
                sample = by_uid.get(record["uid"])
                if sample is None:
                    continue
                item = self.modality.judge_item(spec, task, sample, record)
                if item is not None:
                    items.append(item)
            if not items:
                continue
            owners[owner] = (spec, lang, task, by_uid, records)
            jobs.extend(judge_mod.plan_jobs(
                owner, task, items, self.config.judge,
                raw=self.modality.judge_raw(task)))

        if not jobs:
            return

        for job in jobs:
            print(f"   {job.name:<28} {len(job.items):>6} requests  {job.mode}")

        verdicts_by_owner = await judge_mod.run_jobs(
            jobs, self.config.judge,
            os.path.join(self.config.run_dir, "judge"))

        for owner, verdicts in verdicts_by_owner.items():
            spec, lang, task, by_uid, records = owners[owner]
            for record in records:
                verdict = verdicts.get(record["uid"])
                if verdict is None or str(verdict).strip() == "":
                    continue
                sample = by_uid.get(record["uid"])
                if sample is None:
                    continue
                self.modality.apply_verdict(spec, task, sample, record, verdict)
            write_records(records_path(self.config.run_dir, spec.key, lang),
                          records)

    @classmethod
    def _print_score(cls, lang: str, metric: str, block: Dict[str, Any]) -> None:
        # The block names its own headline; for accuracy-style metrics that is
        # the valid-only figure, which is what gets reported.
        metric = block.get("metric", metric)
        score = block.get(metric)
        if score is None:
            note = block.get("note")
            print(f"   {lang}: no scored items "
                  f"({block.get('unscorable_samples', 0)} unscorable)"
                  + (f" — {note}" if note else ""))
            return

        # ASR and translation report a rate, not a percentage.
        if block.get("display") == "decimal":
            detail = ""
            if block.get("scored_samples") is not None:
                detail = f" ({block['scored_samples']} scored"
                if block.get("extraction_failed"):
                    detail += f", {block['extraction_failed']} empty"
                detail += ")"
            arrow = " (lower is better)" if block.get("lower_is_better") else ""
            print(f"   {lang}: {metric}={score:.4f}{detail}{arrow}")
            cls._print_secondary(block, metric)
            return

        detail = ""
        if "valid_samples" in block:
            correct_valid = round(score * block["valid_samples"])
            detail = f" ({correct_valid}/{block['valid_samples']} valid"
            if block.get("extraction_failed"):
                detail += f", {block['extraction_failed']} no answer"
            if block.get("unscorable_samples"):
                detail += f", {block['unscorable_samples']} unscorable"
            detail += ")"
        elif "scored_samples" in block:
            detail = f" ({block['scored_samples']} scored)"
        arrow = " (lower is better)" if block.get("lower_is_better") else ""
        print(f"   {lang}: {metric}={score * 100:.2f}%{detail}{arrow}")
        cls._print_secondary(block, metric)

    @staticmethod
    def _print_secondary(block: Dict[str, Any], headline: str) -> None:
        """Surface metrics that are not the headline (BLEU beside XCOMET)."""
        extras = [f"{name}={block[name]:.4f}"
                  for name in ("xcomet", "bleu", "chrf2")
                  if name != headline and isinstance(block.get(name),
                                                     (int, float))]
        if extras:
            print(f"          {'  '.join(extras)}")

    @staticmethod
    def _print_groups(block: Dict[str, Any], group_name: str) -> None:
        """Per-group cells, for benchmarks that ask for them."""
        groups = block.get(f"by_{group_name}")
        if not groups:
            return
        metric = block.get("metric", "accuracy")
        width = max(len(name) for name in groups)
        print(f"     by {group_name}:")
        for name, cell in sorted(groups.items()):
            score = cell.get(metric)
            if score is None:
                print(f"       {name.ljust(width)}  no valid answers "
                      f"(n={cell.get('n', 0)})")
                continue
            n_valid, n = cell.get("n_valid", 0), cell.get("n", 0)
            correct_valid = round(score * n_valid)
            note = f" [{n - n_valid} no answer]" if n_valid < n else ""
            print(f"       {name.ljust(width)}  {score * 100:6.2f}%  "
                  f"({correct_valid}/{n_valid} valid){note}")

    # -- driver ------------------------------------------------------------

    async def run(self) -> Dict[str, Dict[str, Any]]:
        specs = self.modality.registry.resolve(
            self.config.benchmarks, self.config.skip_unknown_benchmarks)
        results: Dict[str, Dict[str, Any]] = {}
        if not specs:
            print(f"\nNo {self.modality.name} benchmarks in the selection "
                  f"({', '.join(self.config.benchmarks)}) - nothing to do.")
            await self.client.close()
            return results

        languages = [l for l in LANGUAGE_ORDER if l in self.config.languages]

        # Work is ordered language-major — every Kazakh benchmark, then every
        # English one, then Russian — and split so that everything scored
        # locally finishes before anything waits on the judge API.
        plain: List[tuple] = []
        judged: List[tuple] = []
        for lang in languages:
            for spec in specs:
                if lang not in spec.sources:
                    continue
                (judged if spec.task_for(lang).is_judged else plain).append(
                    (spec, lang))

        results: Dict[str, Dict[str, Any]] = {}
        failures: List[str] = []
        try:
            if plain:
                print(f"\n{'=' * 60}\nscored locally: {len(plain)} "
                      f"benchmark-language pair(s)\n{'=' * 60}")
            for spec, lang in plain:
                try:
                    block = await self._run_pair(spec, lang)
                except Exception as exc:  # noqa: BLE001
                    print(f"!! {spec.key} [{lang}] FAILED: "
                          f"{type(exc).__name__}: {exc}")
                    logger.exception("%s [%s] failed", spec.key, lang)
                    failures.append(f"{spec.key}.{lang}")
                    continue
                if block is not None:
                    results.setdefault(spec.key, {})[lang] = block
                    write_summary(self.config.run_dir, self.config.to_dict(),
                                  {spec.key: results[spec.key]})

            if judged:
                print(f"\n{'=' * 60}\njudged: {len(judged)} "
                      f"benchmark-language pair(s) — generating first\n{'=' * 60}")
                pending = []
                for spec, lang in judged:
                    try:
                        prepared = await self._run_pair(spec, lang,
                                                        defer_judge=True)
                    except Exception as exc:  # noqa: BLE001
                        print(f"!! {spec.key} [{lang}] FAILED: "
                              f"{type(exc).__name__}: {exc}")
                        logger.exception("%s [%s] failed", spec.key, lang)
                        failures.append(f"{spec.key}.{lang}")
                        continue
                    if prepared is not None:
                        pending.append(prepared)

                await self._judge_all(pending)

                for spec, lang, task, samples, records in pending:
                    block = await self._finish_pair(spec, lang, task, samples,
                                                    records)
                    results.setdefault(spec.key, {})[lang] = block
                    write_summary(self.config.run_dir, self.config.to_dict(),
                                  {spec.key: results[spec.key]})
        finally:
            await self.client.close()

        if failures:
            print(f"\n!! {len(failures)} benchmark-language pair(s) failed and "
                  f"were skipped: {', '.join(failures)}")

        return results

    # -- one benchmark-language pair ---------------------------------------

    def _data_for(self, spec: BenchmarkSpec):
        """Load a benchmark once, reuse it for each language.

        Every language is loaded together because alignment intersects uids
        across them; the cache keeps the language-major loop from re-reading
        the same dataset three times.
        """
        if spec.key not in self._data_cache:
            available = [l for l in LANGUAGE_ORDER
                         if l in self.config.languages and l in spec.sources]
            self._data_cache[spec.key] = load_benchmark(
                spec, available, self.config.data_dirs,
                self.config.data_portion, self.config.seed, logger)
        return self._data_cache[spec.key]

    async def _run_pair(self, spec: BenchmarkSpec, lang: str,
                        defer_judge: bool = False):
        """Generate for one benchmark-language pair, and score it unless the
        judge has been deferred to the end of the run."""
        data = self._data_for(spec)
        samples = data.get(lang) or []
        if not samples:
            return None

        task = spec.task_for(lang)
        print(f"\n>> {spec.key} [{lang}] — {len(samples)} items ({task.value})")
        records = await self._evaluate(spec, lang, task, samples)

        if defer_judge:
            return (spec, lang, task, samples, records)
        return await self._finish_pair(spec, lang, task, samples, records)

    async def _finish_pair(self, spec: BenchmarkSpec, lang: str, task: TaskType,
                           samples: List[Sample],
                           records: List[Dict[str, Any]]) -> Dict[str, Any]:
        scored = await self.modality.post_score(spec, task, lang, samples,
                                                records)
        if scored is not records:
            records = scored
        write_records(records_path(self.config.run_dir, spec.key, lang), records)

        metric = self.modality.metric_name(task)
        block = self.modality.aggregate(spec, task, records)
        block["task"] = task.value
        if task.lower_is_better:
            block["lower_is_better"] = True
        if spec.sources[lang].note:
            block["note"] = spec.sources[lang].note
        print(f">> {spec.key} [{lang}]", end="")
        self._print_score(lang, metric, block)
        if spec.print_groups and spec.group_name:
            self._print_groups(block, spec.group_name)
        return block


async def run_evaluation(config: RunConfig,
                         modality: Modality) -> Dict[str, Dict[str, Any]]:
    os.makedirs(config.run_dir, exist_ok=True)
    with open(os.path.join(config.run_dir, "run_config.json"), "w",
              encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, ensure_ascii=False)

    results = await Evaluator(config, modality).run()
    path = write_summary(config.run_dir, config.to_dict(), results)
    print_table(results)
    print(f"Summary: {path}")

    # Rebuilt from the summaries on disk after every pass.
    report = write_performance_report(config.model_dir, config.run_name)
    if report:
        print(f"Report:  {report}")
    return results
