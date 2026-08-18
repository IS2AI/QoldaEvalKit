"""The per-model performance report.

One Markdown document covering every modality, rebuilt from the summary.json
files on disk whenever a pass finishes. The template is fixed, so two models'
reports can be diffed, and a modality that was not run says so rather than
being omitted. Nothing here recomputes a metric.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

LANGUAGES = ("kk", "en", "ru")
LANGUAGE_TITLES = {"kk": "Kazakh", "en": "English", "ru": "Russian"}

# Section order is the order the passes run in.
MODALITIES = ("text", "vision", "audio", "safety", "translation")
MODALITY_TITLES = {
    "text": "Text",
    "vision": "Vision",
    "audio": "Audio",
    "safety": "Safety",
    "translation": "Translation",
}

REPORT_NAME = "performance_report.md"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_score(block: Optional[Dict[str, Any]],
                 metric: Optional[str] = None) -> str:
    """One cell: accuracies as percentages, rates to four decimals."""
    if not block:
        return "–"
    metric = metric or block.get("metric", "accuracy")
    score = block.get(metric)
    if score is None:
        return "n/a"
    if block.get("display") == "decimal":
        text = f"{score:.4f}"
    else:
        text = f"{score * 100:.2f}%"
    if block.get("lower_is_better"):
        text += " ↓"
    return text


def _table(header: List[str], rows: List[List[str]]) -> List[str]:
    if not rows:
        return ["_No results._", ""]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    lines.append("")
    return lines


def _present_languages(results: Dict[str, Any]) -> List[str]:
    return [l for l in LANGUAGES
            if any(l in per_lang for per_lang in results.values())]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _benchmark_table(results: Dict[str, Any]) -> List[str]:
    """Benchmarks down, languages across — the terminal table, in Markdown."""
    langs = _present_languages(results)
    if not langs:
        return ["_No results._", ""]
    header = ["Benchmark"] + [LANGUAGE_TITLES[l] for l in langs] + ["Metric"]
    rows = []
    for name, per_lang in results.items():
        block = next((per_lang[l] for l in langs if l in per_lang), None)
        metric = (block or {}).get("metric", "accuracy")
        rows.append([f"`{name}`"]
                    + [format_score(per_lang.get(l)) for l in langs]
                    + [f"`{metric}`"])
    return _table(header, rows)


def _coverage_table(results: Dict[str, Any]) -> List[str]:
    """How many items each score rests on, and what fell out of it."""
    langs = _present_languages(results)
    rows = []
    for name, per_lang in results.items():
        for lang in langs:
            block = per_lang.get(lang)
            if not block:
                continue
            rows.append([
                f"`{name}`", LANGUAGE_TITLES[lang],
                str(block.get("total_samples", "–")),
                str(block.get("scored_samples", "–")),
                str(block.get("extraction_failed", 0)),
                str(block.get("unscorable_samples", 0)),
            ])
    return _table(["Benchmark", "Language", "Items", "Scored",
                   "No answer", "Unscorable"], rows)


def _group_table(results: Dict[str, Any], benchmark: str,
                 group_name: str, title: str) -> List[str]:
    """Per-category cells for one benchmark, languages across."""
    per_lang = results.get(benchmark)
    if not per_lang:
        return []
    langs = [l for l in LANGUAGES if l in per_lang]
    key = f"by_{group_name}"
    names: List[str] = []
    for lang in langs:
        for name in (per_lang[lang].get(key) or {}):
            if name not in names:
                names.append(name)
    if not names:
        return []

    metric = per_lang[langs[0]].get("metric", "accuracy")
    rows = []
    for name in sorted(names):
        row = [f"`{name}`"]
        for lang in langs:
            cell = (per_lang[lang].get(key) or {}).get(name)
            if not cell:
                row.append("–")
                continue
            value = cell.get(metric, cell.get("accuracy"))
            row.append("n/a" if value is None else f"{value * 100:.2f}%"
                       + f" ({cell.get('n', 0)})")
        rows.append(row)
    return ([f"#### {title}", ""]
            + _table([group_name.replace("_", " ")]
                     + [LANGUAGE_TITLES[l] for l in langs], rows))


def _translation_table(results: Dict[str, Any]) -> List[str]:
    """Every direction with all three metrics side by side."""
    rows = []
    for name, per_lang in results.items():
        for lang, block in per_lang.items():
            rows.append([
                f"`{name}`",
                format_score(block, "xcomet"),
                format_score(block, "bleu"),
                format_score(block, "chrf2"),
                str(block.get("scored_samples", "–")),
            ])
    return _table(["Direction", "XCOMET", "BLEU", "chrF++", "Scored"], rows)


def _safety_section(results: Dict[str, Any]) -> List[str]:
    lines = _benchmark_table(results)
    lines += _group_table(results, "qorgau", "risk_area", "By risk area")
    return lines


def _audio_section(results: Dict[str, Any]) -> List[str]:
    lines = _benchmark_table(results)
    lines += _group_table(results, "sakura", "category_hop",
                          "SAKURA — attribute x hop")
    lines += _group_table(results, "spokenmqa", "subset",
                          "SpokenMQA — subset")
    return lines


SECTION_BUILDERS = {
    "audio": _audio_section,
    "safety": _safety_section,
    "translation": _translation_table,
}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _load(model_dir: str, modality: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    path = os.path.join(model_dir, modality, "summary.json")
    if not os.path.exists(path):
        return {}, {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}, {}
    return summary.get("results") or {}, summary.get("run") or {}


def _settings_block(run: Dict[str, Any]) -> List[str]:
    if not run:
        return []
    sampling = run.get("sampling") or {}
    model = run.get("model") or {}
    judge = run.get("judge") or {}
    rows = [
        ["Model", f"`{model.get('model_name', '–')}`"],
        ["Data portion", str(run.get("data_portion", "–"))],
        ["Thinking", str(model.get("thinking", "–"))],
        ["Temperature", str(sampling.get("temperature", "–"))],
        ["top_p / top_k", f"{sampling.get('top_p', '–')} / "
                          f"{sampling.get('top_k', '–')}"],
        ["Presence penalty", str(sampling.get("presence_penalty", "–"))],
        ["Max tokens", str(sampling.get("max_tokens", "–"))],
        ["Judge", f"`{judge.get('model', '–')}` "
                  f"(effort {judge.get('reasoning_effort', '–')})"],
    ]
    return ["## Run settings", ""] + _table(["Setting", "Value"], rows)


def build_report(model_dir: str, run_name: str) -> str:
    """The whole document as a string."""
    lines = [f"# Performance report — {run_name}", "",
             f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} from "
             f"`{os.path.basename(model_dir)}/*/summary.json`._", ""]

    first_run: Dict[str, Any] = {}
    sections: List[Tuple[str, Dict[str, Any]]] = []
    for modality in MODALITIES:
        results, run = _load(model_dir, modality)
        sections.append((modality, results))
        if run and not first_run:
            first_run = run

    done = [MODALITY_TITLES[m] for m, r in sections if r]
    missing = [MODALITY_TITLES[m] for m, r in sections if not r]
    lines += ["**Evaluated:** " + (", ".join(done) if done else "nothing yet")]
    if missing:
        lines += ["", "**Not run:** " + ", ".join(missing)]
    lines += [""]

    lines += _settings_block(first_run)

    for modality, results in sections:
        lines += [f"## {MODALITY_TITLES[modality]}", ""]
        if not results:
            lines += ["_Not run._", ""]
            continue
        builder = SECTION_BUILDERS.get(modality)
        lines += builder(results) if builder else _benchmark_table(results)
        if modality not in ("translation",):
            lines += ["<details><summary>Coverage</summary>", ""]
            lines += _coverage_table(results)
            lines += ["</details>", ""]

    lines += ["---", "",
              "Accuracies are percentages over the items an answer could be "
              "parsed from (`accuracy_valid_only`). WER, XCOMET, BLEU and "
              "chrF++ are rates in 0–1. ↓ marks a metric where lower is "
              "better.", ""]
    return "\n".join(lines)


def write_performance_report(model_dir: str, run_name: str) -> Optional[str]:
    """Rebuild the report from whatever summaries exist. Never fatal."""
    try:
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, REPORT_NAME)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(build_report(model_dir, run_name))
        return path
    except OSError:
        return None
