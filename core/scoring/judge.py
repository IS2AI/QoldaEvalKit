"""LLM-as-a-judge over the OpenAI Batch API.

One batch per benchmark and language.  Each request carries only what is
strictly needed to decide correctness — the model response plus the gold answer
(plus the instruction and its constraints for IFBench, which has no gold
answer) — and the judge is constrained to emit a single ``0`` or ``1``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..config import JudgeConfig
from ..registry import TaskType

logger = logging.getLogger("qoldaevalkit")

# OpenAI's per-batch request ceiling; larger jobs are split transparently.
MAX_REQUESTS_PER_BATCH = 50_000

TERMINAL_STATES = {"completed", "failed", "expired", "cancelled"}

JUDGE_SYSTEM = (
    "You grade whether a model's response means the same thing as the correct "
    "answer. Judge meaning, not wording.\n"
    "Answer 1 if the response conveys the same answer as the correct answer, "
    "even when it differs in wording, word order, level of detail, "
    "formatting, punctuation, capitalisation, or language, and even when it "
    "is embedded in a longer sentence or adds correct extra context.\n"
    "Answer 0 only if the response states something different from the "
    "correct answer, contradicts it, omits the part that actually answers "
    "the question, or does not answer at all.\n"
    "Reply with the single digit 1 or 0 and nothing else."
)

IF_JUDGE_SYSTEM = (
    "You grade instruction following. Answer 1 if the response satisfies "
    "every listed constraint, 0 if it violates any of them.\n"
    "Judge only the constraints listed. Whether the response is factually "
    "correct or well written is irrelevant here — a wrong but compliant "
    "answer scores 1.\n"
    "Reply with the single digit 1 or 0 and nothing else."
)


@dataclass
class JudgeItem:
    """One thing to grade.

    Most benchmarks let the templates below build the messages. A benchmark
    with its own published rubric (Qorgau) sets ``system``/``user`` instead and
    those are sent verbatim, so the graded prompt matches the original work.
    """

    uid: str
    response: str
    reference: Optional[str] = None
    instruction: Optional[str] = None
    constraints: Optional[str] = None
    # Verbatim messages; when set they replace the templated ones.
    system: Optional[str] = None
    user: Optional[str] = None
    # Overrides JudgeConfig.max_completion_tokens for this item. A rubric needs
    # room for a numbered list, not the single character a 0/1 verdict needs.
    max_completion_tokens: Optional[int] = None


def build_messages(task: TaskType, item: JudgeItem) -> List[Dict[str, str]]:
    if item.user is not None:
        messages: List[Dict[str, str]] = []
        if item.system:
            messages.append({"role": "system", "content": item.system})
        messages.append({"role": "user", "content": item.user})
        return messages

    if task == TaskType.IF_JUDGE:
        user = (
            f"Instruction:\n{item.instruction}\n\n"
            f"Constraints:\n{item.constraints}\n\n"
            f"Model response:\n{item.response}"
        )
        return [{"role": "system", "content": IF_JUDGE_SYSTEM},
                {"role": "user", "content": user}]

    user = (
        f"Correct answer:\n{item.reference}\n\n"
        f"Model response:\n{item.response}"
    )
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Structured outputs — a strict schema the judge cannot deviate from
# ---------------------------------------------------------------------------

# strict + additionalProperties:false: the reply is exactly {"verdict": 0|1}.
VERDICT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"verdict": {"type": "integer", "enum": [0, 1]}},
            "required": ["verdict"],
            "additionalProperties": False,
        },
    },
}

# Qorgau's rubric: one yes/no per question, in order.
RUBRIC_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "rubric_answers",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answers": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["yes", "no"]},
                },
            },
            "required": ["answers"],
            "additionalProperties": False,
        },
    },
}


def _schema_for(task: TaskType):
    return RUBRIC_SCHEMA if task == TaskType.SAFETY_RUBRIC else VERDICT_SCHEMA


def _request_body(task: TaskType, item: JudgeItem,
                  cfg: JudgeConfig) -> Dict[str, object]:
    tokens = item.max_completion_tokens or cfg.max_completion_tokens
    body: Dict[str, object] = {
        "model": cfg.model,
        "messages": build_messages(task, item),
    }
    if cfg.structured:
        body["response_format"] = _schema_for(task)
        # {"verdict": 1} needs more room than a bare "1" does.
        tokens = max(tokens, 16)
    body["max_completion_tokens"] = tokens
    if cfg.reasoning_effort:
        body["reasoning_effort"] = cfg.reasoning_effort
    return body


def _parse_verdict(content: str) -> Optional[int]:
    """A 0/1 verdict, from either the schema's JSON or a bare digit.

    Both shapes are accepted so the parser works with structured outputs on or
    off, and so results written by an earlier run still read correctly.
    """
    text = (content or "").strip()
    if not text:
        return None

    if text.startswith("{"):
        try:
            value = json.loads(text).get("verdict")
        except json.JSONDecodeError:
            value = None
        if value in (0, 1):
            return int(value)
        if isinstance(value, str) and value.strip() in ("0", "1"):
            return int(value.strip())

    for ch in text:
        if ch in "01":
            return int(ch)
        if not ch.isspace():
            break
    return None


async def _run_batch(name: str, task: TaskType, items: List[JudgeItem],
                     cfg: JudgeConfig, workdir: str,
                     raw: bool = False) -> Dict[str, Optional[object]]:
    """Submit ``items`` as one batch job, block until done, return uid -> verdict.

    The verdict is 0/1 by default. With ``raw=True`` it is the judge's reply
    text, which is what a rubric needs — the caller then parses the numbered
    answers itself.

    A uid missing from the result (or mapped to ``None``) was not gradeable and
    is excluded from the benchmark's denominator by the caller.
    """
    if not items:
        return {}
    if not cfg.enabled:
        logger.info("%s: judging disabled, responses saved only", name)
        return {}
    if not cfg.api_key:
        logger.warning("%s: no judge API key (JUDGE_API_KEY / OPENAI_API_KEY); "
                       "skipping judging", name)
        return {}

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    os.makedirs(workdir, exist_ok=True)
    verdicts: Dict[str, Optional[int]] = {}

    chunks = [items[i:i + MAX_REQUESTS_PER_BATCH]
              for i in range(0, len(items), MAX_REQUESTS_PER_BATCH)]

    try:
        for chunk_index, chunk in enumerate(chunks):
            suffix = "" if len(chunks) == 1 else f".part{chunk_index}"
            request_path = os.path.join(workdir, f"{name}{suffix}.requests.jsonl")
            with open(request_path, "w", encoding="utf-8") as handle:
                for item in chunk:
                    handle.write(json.dumps({
                        "custom_id": item.uid,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": _request_body(task, item, cfg),
                    }, ensure_ascii=False) + "\n")

            logger.info("%s: uploading %d judge requests", name, len(chunk))
            with open(request_path, "rb") as handle:
                uploaded = await client.files.create(file=handle, purpose="batch")

            batch = await client.batches.create(
                input_file_id=uploaded.id,
                endpoint="/v1/chat/completions",
                completion_window=cfg.completion_window,
                metadata={"benchmark": name},
            )
            logger.info("%s: batch %s submitted (%s)", name, batch.id, cfg.model)
            print(f"  [judge] {name}: batch {batch.id} submitted "
                  f"({len(chunk)} requests) — waiting…")

            batch = await _wait_for_batch(client, batch.id, name, cfg)

            if batch.status != "completed":
                detail = getattr(getattr(batch, "errors", None), "data", None)
                logger.error("%s: batch %s ended as %s (%s)",
                             name, batch.id, batch.status, detail)
                print(f"  [judge] {name}: batch {batch.id} {batch.status}")
                if not batch.output_file_id:
                    continue

            output_path = os.path.join(workdir, f"{name}{suffix}.results.jsonl")
            content = await client.files.content(batch.output_file_id)
            # Not named `raw`: that is the flag choosing the verdict shape.
            payload = content.read()
            with open(output_path, "wb") as handle:
                handle.write(payload)

            for line in payload.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                uid = record.get("custom_id")
                try:
                    message = (record["response"]["body"]["choices"][0]
                               ["message"]["content"])
                except (KeyError, IndexError, TypeError):
                    verdicts[uid] = None
                    continue
                verdicts[uid] = message if raw else _parse_verdict(message)
    finally:
        await client.close()

    graded = sum(1 for v in verdicts.values()
                 if v is not None and str(v).strip() != "")
    logger.info("%s: %d/%d verdicts returned", name, graded, len(items))
    return verdicts


async def _wait_for_batch(client, batch_id: str, name: str, cfg: JudgeConfig):
    """Poll until the batch reaches a terminal state or the timeout expires."""
    started = time.monotonic()
    last_counts = None
    while True:
        batch = await client.batches.retrieve(batch_id)
        if batch.status in TERMINAL_STATES:
            return batch

        counts = getattr(batch, "request_counts", None)
        if counts is not None and (counts.completed, counts.failed) != last_counts:
            last_counts = (counts.completed, counts.failed)
            print(f"  [judge] {name}: {batch.status} — "
                  f"{counts.completed}/{counts.total} done, "
                  f"{counts.failed} failed")

        if time.monotonic() - started > cfg.timeout:
            logger.error("%s: batch %s timed out after %.0fs (status=%s)",
                         name, batch_id, cfg.timeout, batch.status)
            return batch

        await asyncio.sleep(cfg.poll_interval)


# ---------------------------------------------------------------------------
# Job planning: small sets go direct, large ones are batched (and split)
# ---------------------------------------------------------------------------

@dataclass
class JudgeJob:
    """One unit of grading work — a whole set, or one slice of a large one."""

    owner: str                 # "<benchmark>.<lang>", where verdicts belong
    name: str                  # job label, unique per run
    task: TaskType
    items: List[JudgeItem]
    mode: str                  # "direct" | "batch"
    raw: bool = False


def plan_jobs(owner: str, task: TaskType, items: List[JudgeItem],
              cfg: JudgeConfig, raw: bool = False) -> List[JudgeJob]:
    """Decide how one benchmark-language's grading is carried out.

    Under ``direct_below`` the Batch API is not worth its queue wait, so the
    requests go through the chat API concurrently. Above ``split_above`` the
    set is cut into ``split_parts`` jobs so the enqueued-token cost is spread
    over several submissions rather than one very large one.
    """
    if not items:
        return []

    if len(items) < cfg.direct_below:
        return [JudgeJob(owner, owner, task, list(items), "direct", raw)]

    if len(items) > cfg.split_above and cfg.split_parts > 1:
        parts = cfg.split_parts
        size = -(-len(items) // parts)          # ceil, so the last part is the short one
        return [
            JudgeJob(owner, f"{owner}.part{i + 1}", task,
                     items[i * size:(i + 1) * size], "batch", raw)
            for i in range(parts)
            if items[i * size:(i + 1) * size]
        ]

    return [JudgeJob(owner, owner, task, list(items), "batch", raw)]


def _retry_after(exc: Exception) -> Optional[float]:
    """The server's own Retry-After, in seconds, when it sent one."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    for name in ("retry-after", "retry-after-ms", "x-ratelimit-reset-requests"):
        raw = headers.get(name)
        if not raw:
            continue
        try:
            value = float(str(raw).rstrip("smh"))
        except ValueError:
            continue
        if name.endswith("-ms"):
            value /= 1000.0
        return min(value, 120.0)
    return None


async def _run_direct(job: JudgeJob, cfg: JudgeConfig, workdir: str,
                      gate: Optional[asyncio.Semaphore] = None
                      ) -> Dict[str, Optional[object]]:
    """Grade a small set through the chat API.

    ``gate`` is shared by every direct job, so ``direct_concurrency`` caps
    judge requests in flight in total rather than per job.
    """
    import random

    from openai import AsyncOpenAI

    # SDK retry off: one backoff policy, not two nested ones.
    client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url,
                         max_retries=0)
    semaphore = gate or asyncio.Semaphore(max(1, cfg.direct_concurrency))
    verdicts: Dict[str, Optional[object]] = {}

    async def one(item: JudgeItem) -> None:
        async with semaphore:
            body = _request_body(job.task, item, cfg)
            last: Optional[Exception] = None
            for attempt in range(cfg.direct_max_attempts):
                try:
                    response = await client.chat.completions.create(**body)
                    content = response.choices[0].message.content
                    verdicts[item.uid] = (content if job.raw
                                          else _parse_verdict(content or ""))
                    return
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    if attempt == cfg.direct_max_attempts - 1:
                        break
                    # Jitter stops a rate-limited burst waking in lockstep.
                    delay = _retry_after(exc)
                    if delay is None:
                        delay = min(2.0 ** attempt, 60.0) * (1 + random.random())
                    await asyncio.sleep(delay)
            logger.warning("%s: %s failed to grade after %d attempts (%s)",
                           job.name, item.uid, cfg.direct_max_attempts, last)
            verdicts[item.uid] = None

    mode = ("sequentially" if cfg.direct_concurrency <= 1
            else f"{cfg.direct_concurrency} at a time")
    print(f"  [judge] {job.name}: {len(job.items)} requests direct, {mode} "
          f"(under the {cfg.direct_below} batching threshold)")
    try:
        await asyncio.gather(*[one(i) for i in job.items])
    finally:
        await client.close()

    ungraded = sum(1 for item in job.items if verdicts.get(item.uid) is None)
    if ungraded:
        # Loud: these items drop out of the denominator.
        print(f"  [judge] {job.name}: WARNING {ungraded} of {len(job.items)} "
              f"could not be graded — they will count as unscorable. "
              f"Re-run with --resume to retry just these.")

    os.makedirs(workdir, exist_ok=True)
    with open(os.path.join(workdir, f"{job.name}.direct.jsonl"), "w",
              encoding="utf-8") as handle:
        for item in job.items:
            handle.write(json.dumps({"custom_id": item.uid,
                                     "verdict": verdicts.get(item.uid)},
                                    ensure_ascii=False) + "\n")
    return verdicts


async def run_jobs(jobs: List[JudgeJob], cfg: JudgeConfig,
                   workdir: str) -> Dict[str, Dict[str, Optional[object]]]:
    """Run every grading job, at most ``cfg.max_parallel`` batches at a time.

    Returns verdicts grouped by owner, so a benchmark split across several jobs
    gets its slices merged back together.
    """
    results: Dict[str, Dict[str, Optional[object]]] = {}
    if not jobs:
        return results
    if not cfg.enabled:
        logger.info("judging disabled; responses saved only")
        return results
    if not cfg.api_key:
        logger.warning("no judge API key (JUDGE_API_KEY / OPENAI_API_KEY); "
                       "skipping %d grading job(s)", len(jobs))
        return results

    gate = asyncio.Semaphore(max(1, cfg.max_parallel))
    # Shared by every direct job: a run-wide cap, not per job.
    direct_gate = asyncio.Semaphore(max(1, cfg.direct_concurrency))

    async def run_one(job: JudgeJob) -> None:
        if job.mode == "direct":
            verdicts = await _run_direct(job, cfg, workdir, direct_gate)
        else:
            async with gate:     # only batch jobs consume the queue budget
                verdicts = await _run_batch(job.name, job.task, job.items,
                                            cfg, workdir, job.raw)
        results.setdefault(job.owner, {}).update(verdicts)

    batches = sum(1 for j in jobs if j.mode == "batch")
    direct = len(jobs) - batches
    how = ("one at a time" if cfg.direct_concurrency <= 1
           else f"{cfg.direct_concurrency} requests at a time, shared")
    print(f"\n>> grading: {len(jobs)} job(s) — {batches} batched "
          f"({cfg.max_parallel} at a time), {direct} direct ({how})")
    await asyncio.gather(*[run_one(j) for j in jobs])
    return results
