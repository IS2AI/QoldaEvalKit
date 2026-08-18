"""Per-item record files, metric aggregation and the run summary."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger("qoldaevalkit")


def setup_logging(run_dir: str, verbose: bool = False) -> logging.Logger:
    os.makedirs(run_dir, exist_ok=True)
    log = logging.getLogger("qoldaevalkit")
    log.setLevel(logging.INFO)
    log.handlers.clear()

    file_handler = logging.FileHandler(os.path.join(run_dir, "eval.log"),
                                       mode="a", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
    )
    file_handler.setLevel(logging.INFO)
    log.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("  %(levelname)s: %(message)s"))
    stream_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    log.addHandler(stream_handler)
    return log


def records_path(run_dir: str, benchmark: str, lang: str) -> str:
    return os.path.join(run_dir, "records", f"{benchmark}.{lang}.jsonl")


def load_records(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed record in %s", path)
    return records


def write_records(path: str, records: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def has_prediction(record: Dict[str, Any]) -> bool:
    """Did an answer come back that could be read at all?

    The single definition of "extraction succeeded", shared by the metric
    blocks and the runner's retry pass.
    """
    prediction = record.get("prediction")
    if prediction is None:
        return False
    return str(prediction).strip() != ""


def aggregate(records: List[Dict[str, Any]], metric_name: str,
              group_name: Optional[str] = None) -> Dict[str, Any]:
    """Turn per-item records into a metric block.

    Two things are reported separately rather than folded into the score:

    * ``correct is None`` — the item could not be graded at all (symbolic check
      undecided, judge verdict missing). Excluded from the denominator, so the
      score is never quietly depressed by grader failures.
    * ``prediction`` empty — the model produced no parseable answer. This is a
      format/extraction problem rather than a wrong answer, so the reported
      figure is ``<metric>_valid_only``, over the items an answer could be read
      from. ``<metric>`` keeps the all-items number alongside it, and
      ``extraction_failed`` says how many were dropped — if that count is high,
      the reported score is standing on a small denominator.

    None of this depends on ``--save_responses``: it is computed from the
    ``correct`` and ``prediction`` fields, which are always written.
    """
    total = len(records)
    scored = [r for r in records if r.get("correct") is not None]
    correct = sum(1 for r in scored if r["correct"])
    unscorable = total - len(scored)
    truncated = sum(1 for r in records if r.get("truncated_thinking"))

    _extracted = has_prediction

    valid = [r for r in scored if _extracted(r)]
    valid_correct = sum(1 for r in valid if r["correct"])
    extraction_failed = total - sum(1 for r in records if _extracted(r))

    valid_name = f"{metric_name}_valid_only"
    block: Dict[str, Any] = {
        # Headline is the valid-only figure; the all-items figure stays
        # alongside under `<metric>`.
        "metric": valid_name,
        metric_name: round(correct / len(scored), 6) if scored else None,
        f"{metric_name}_pct": (f"{correct / len(scored) * 100:.2f}%"
                               if scored else None),
        valid_name: (round(valid_correct / len(valid), 6) if valid else None),
        f"{valid_name}_pct": (f"{valid_correct / len(valid) * 100:.2f}%"
                              if valid else None),
        "total_samples": total,
        "scored_samples": len(scored),
        "correct_samples": correct,
        "extraction_failed": extraction_failed,
        "valid_samples": len(valid),
        "unscorable_samples": unscorable,
        "truncated_thinking_samples": truncated,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if group_name:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for record in scored:
            groups.setdefault(str(record.get("group", "unknown")), []).append(record)
        block[f"by_{group_name}"] = {}
        for name, items in sorted(groups.items()):
            good = [r for r in items if _extracted(r)]
            block[f"by_{group_name}"][name] = {
                metric_name: round(sum(1 for r in items if r["correct"]) / len(items), 6),
                valid_name: (round(sum(1 for r in good if r["correct"]) / len(good), 6)
                             if good else None),
                "n": len(items),
                "n_valid": len(good),
                "correct": sum(1 for r in items if r["correct"]),
            }
    return block


def write_summary(run_dir: str, run_meta: Dict[str, Any],
                  results: Dict[str, Dict[str, Any]]) -> str:
    """Merge into any existing summary so partial reruns accumulate."""
    path = os.path.join(run_dir, "summary.json")
    summary: Dict[str, Any] = {"run": run_meta, "results": {}}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                summary = json.load(handle)
        except json.JSONDecodeError:
            pass
        summary["run"] = run_meta
    summary.setdefault("results", {})

    for benchmark, per_language in results.items():
        summary["results"].setdefault(benchmark, {}).update(per_language)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return path


def print_table(results: Dict[str, Dict[str, Any]]) -> None:
    """Compact benchmark x language score table."""
    languages = ["kk", "en", "ru"]     # the order benchmarks are run in
    present = [l for l in languages
               if any(l in per_lang for per_lang in results.values())]
    if not present:
        return

    width = max([len(k) for k in results] + [9])
    header = "Benchmark".ljust(width) + "".join(f"{l.upper():>12}" for l in present)
    print("\n" + header)
    print("-" * len(header))
    for benchmark, per_language in results.items():
        row = benchmark.ljust(width)
        for lang in present:
            block = per_language.get(lang)
            if not block:
                cell = "-"
            else:
                metric = block.get("metric", "accuracy")
                score = block.get(metric)
                if score is None:
                    cell = "n/a"
                elif block.get("display") == "decimal":
                    # WER, XCOMET, BLEU: a rate, shown as 0.1707.
                    cell = f"{score:.4f}"
                    if block.get("lower_is_better"):
                        cell += "\u2193"
                else:
                    cell = f"{score * 100:.2f}"
                    if block.get("lower_is_better"):
                        cell += "\u2193"
            row += f"{cell:>12}"
        print(row)
    print()
