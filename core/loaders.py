"""Dataset loading, cross-lingual alignment and deterministic subsampling."""

from __future__ import annotations

import json
import logging
import os
import random
import re
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional

from .registry import BenchmarkSpec, Sample, Source

_NUM = re.compile(r"(\d+)")


def _natural_key(text: str):
    """Sort '9' before '10' so subsampling is stable and human-legible."""
    return tuple(
        int(part) if part.isdigit() else part
        for part in _NUM.split(str(text))
    )


def _hf_token(public: bool) -> Optional[str]:
    """The token is always sent when present (rate limits), but only required
    for the gated and private datasets."""
    token = os.getenv("HF_TOKEN")
    if not token and not public:
        raise RuntimeError(
            "HF_TOKEN is required for gated/private datasets. "
            "Put it in .env or export it before running."
        )
    return token or None


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# Blank spreadsheet cells arrive as float NaN and, once a column has been read
# as str, can stringify to "nan" — which would otherwise become a prompt.
_BLANK_CELLS = {"", "nan", "none", "nat", "<na>"}


def _load_csv(path: str) -> List[Dict[str, Any]]:
    """A CSV as row dicts, with blank cells as None."""
    import pandas as pd

    frame = pd.read_csv(path, dtype=str)

    def clean(value: Any) -> Any:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).strip()
        return None if text.lower() in _BLANK_CELLS else text

    return [{key: clean(value) for key, value in row.items()}
            for row in frame.to_dict(orient="records")]


def _load_xlsx(path: str, sheet: Optional[str]) -> List[Dict[str, Any]]:
    """One sheet of a workbook as row dicts, with blank cells as None."""
    import pandas as pd

    frame = pd.read_excel(path, sheet_name=sheet or 0, dtype=str)

    def clean(value: Any) -> Any:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).strip()
        return None if text.lower() in _BLANK_CELLS else text

    return [{key: clean(value) for key, value in row.items()}
            for row in frame.to_dict(orient="records")]


def _resolve_configs(source: Source, logger: logging.Logger) -> Source:
    """Replace a hardcoded config list with the Hub's, when available."""
    if not source.configs_from_hub:
        return source
    try:
        from datasets import get_dataset_config_names

        names = get_dataset_config_names(
            source.path, token=_hf_token(source.public)
        )
        if names:
            source.configs = list(names)
            logger.info("%s: discovered %d configs on the Hub",
                        source.path, len(names))
    except Exception as exc:  # noqa: BLE001 - fall back to the static list
        logger.warning("%s: config discovery failed (%s); using the built-in list",
                       source.path, exc)
    return source


_MANIFEST_AUDIO_KEYS = ("audio", "audio_path", "file_name", "filename",
                        "path", "file", "wav")
_MANIFEST_TEXT_KEYS = ("text", "raw_transcription", "transcription",
                       "sentence", "reference", "target")


def _find_manifest(root: str, config: Optional[str],
                   patterns: Optional[List[str]] = None) -> Optional[str]:
    """Locate a manifest for one language under a user-supplied ASR corpus.

    ``patterns`` from the spec are tried first (they name the files a given
    corpus actually ships), then a generic search so that other layouts work
    without anyone having to rearrange their download:

        <root>/<config>/{test,dev,manifest}.{tsv,csv,jsonl}
        <root>/<config>.{tsv,csv,jsonl}
        <root>/{test,manifest}.{tsv,csv,jsonl}     (single-language corpus)
    """
    import glob

    names = ["test", "dev", "validation", "manifest", "data"]
    extensions = ["tsv", "csv", "jsonl"]
    candidates: List[str] = [os.path.join(root, p) for p in (patterns or [])]

    if config:
        for name in names:
            candidates += [os.path.join(root, config, f"{name}.{e}")
                           for e in extensions]
        candidates += [os.path.join(root, config, f"*.{e}") for e in extensions]
        candidates += [os.path.join(root, f"{config}.{e}") for e in extensions]
        # FLEURS configs look like "kk_kz"; the older harness used "kaz".
        candidates += [os.path.join(root, f"*{config}*.tsv"),
                       os.path.join(root, f"*{config}*.jsonl")]
    for name in names:
        candidates += [os.path.join(root, f"{name}.{e}") for e in extensions]

    for pattern in candidates:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def _read_manifest(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        return _load_jsonl(path)
    import csv as _csv

    delimiter = "\t" if path.endswith(".tsv") else ","
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        # A FLEURS TSV ships without a header row; detect that and name the
        # columns positionally the way FLEURS orders them.  The test is on
        # whole fields, not substrings: a data row holding "u0.wav" must not
        # be mistaken for a header just because it contains "wav".
        first_fields = {field.strip().strip('"').lower()
                        for field in sample.split("\n")[0].split(delimiter)}
        has_header = bool(first_fields
                          & set(_MANIFEST_AUDIO_KEYS + _MANIFEST_TEXT_KEYS))
        if has_header:
            return [dict(row) for row in _csv.DictReader(handle, delimiter=delimiter)]
        fields = ["id", "file_name", "raw_transcription", "transcription",
                  "phonemes", "num_samples", "gender"]
        rows = []
        for parts in _csv.reader(handle, delimiter=delimiter):
            if not parts:
                continue
            rows.append({fields[i]: parts[i]
                         for i in range(min(len(fields), len(parts)))})
        return rows


def _resolve_audio_path(root: str, manifest_path: str, value: Any) -> Optional[str]:
    name = str(value or "").strip()
    if not name:
        return None
    if os.path.isabs(name) and os.path.exists(name):
        return name
    manifest_dir = os.path.dirname(manifest_path)
    for base in (manifest_dir, root):
        for prefix in ("", "audio", "wavs", "clips",
                       os.path.join("audio", "test"), "test"):
            candidate = os.path.join(base, prefix, name) if prefix else os.path.join(base, name)
            if os.path.exists(candidate):
                return candidate
    return None


def _load_manifest_rows(root: str, config: Optional[str],
                        logger: logging.Logger,
                        patterns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Rows from a local ASR corpus, each carrying a resolved audio path."""
    if not root or not os.path.isdir(root):
        return []
    manifest = _find_manifest(root, config, patterns)
    if not manifest:
        logger.info("No ASR manifest for '%s' under %s", config, root)
        return []

    rows: List[Dict[str, Any]] = []
    missing = 0
    for row in _read_manifest(manifest):
        audio_value = next((row[k] for k in _MANIFEST_AUDIO_KEYS
                            if row.get(k)), None)
        text_value = next((row[k] for k in _MANIFEST_TEXT_KEYS if row.get(k)), None)
        path = _resolve_audio_path(root, manifest, audio_value)
        if not path or not text_value:
            missing += 1
            continue
        row = dict(row)
        row["__audio_path__"] = path
        row.setdefault("text", text_value)
        rows.append(row)

    logger.info("ASR manifest %s: %d usable rows (%d skipped)",
                manifest, len(rows), missing)
    return rows


def _path_thunk(path: str) -> Callable[[], Any]:
    def load() -> Any:
        return path
    return load


def _media_thunk(dataset, index: int, column: str) -> Callable[[], Any]:
    """Defer decoding one image / clip until the request that needs it."""
    def load() -> Any:
        return dataset[index][column]
    return load


def _pick_column(dataset, candidates: Optional[List[str]],
                 what: str, source: Source, config, split,
                 logger: logging.Logger) -> Optional[str]:
    if not candidates:
        return None
    column = next((c for c in candidates if c in dataset.column_names), None)
    if column is None:
        logger.warning("%s [%s/%s]: none of the %s columns %s are present "
                       "(have %s)", source.path, config, split, what,
                       candidates, dataset.column_names)
    return column


def _ensure_unique_uids(samples: List[Sample], source: Source,
                        logger: logging.Logger) -> List[Sample]:
    """Make every uid unique, appending an occurrence suffix to repeats.

    A uid has to identify exactly one item: the runner keys records by it, and
    it becomes the ``custom_id`` of a judge request — which the Batch API
    rejects outright if two requests share one.  Not every dataset's own id is
    unique (Qorgau reuses an id across its question types), so this guards the
    invariant rather than trusting the source.

    Deterministic: repeats are numbered in file order, so parallel languages
    that share a duplication pattern still line up.
    """
    seen: Counter = Counter()
    duplicates = 0
    for sample in samples:
        seen[sample.uid] += 1
        occurrence = seen[sample.uid]
        if occurrence > 1:
            sample.uid = f"{sample.uid}~{occurrence}"
            duplicates += 1
    if duplicates:
        logger.warning("%s: %d duplicate uid(s) made unique with a ~n suffix",
                       source.path, duplicates)
    return samples


def load_source(source: Source, data_dirs: Dict[str, str],
                logger: logging.Logger) -> List[Sample]:
    """Materialise one language's data for a benchmark as normalised samples."""
    source = _resolve_configs(source, logger)
    data_dir = data_dirs.get(source.dir_key, ".")
    suppress_fallback = False
    samples: List[Sample] = []
    counters: Counter = Counter()
    global_index = 0

    for config, split in source.parts():
        dataset = None
        image_column = None
        audio_column = None
        manifest_rows: Optional[List[Dict[str, Any]]] = None

        if source.kind == "manifest":
            root = source.path or ""
            if root and not os.path.isabs(root):
                root = os.path.join(data_dir, root)
            root = root or data_dir
            corpus_provided = bool(root) and os.path.isdir(root)
            manifest_rows = (_load_manifest_rows(root, config, logger,
                                                 source.manifest_patterns)
                             if corpus_provided else [])
            if not manifest_rows:
                if corpus_provided:
                    # Someone pointed us at a corpus and it did not work out.
                    # Quietly downloading gigabytes from the Hub instead would
                    # hide their mistake, so this is an error, not a fallback.
                    logger.error(
                        "No usable ASR data for '%s' under %s. Expected a "
                        "manifest such as %s/%s/test.tsv (or .csv/.jsonl) with "
                        "an audio column and a text column, plus the audio "
                        "files it references.", config, root, root, config)
                    suppress_fallback = True
                continue
            rows: Iterable[Dict[str, Any]] = manifest_rows
        elif source.kind == "csv":
            path = source.path
            if not os.path.isabs(path):
                path = os.path.join(data_dir, path)
            if not os.path.exists(path):
                logger.warning("Missing local dataset: %s", path)
                return []
            rows = _load_csv(path)
        elif source.kind == "xlsx":
            path = source.path
            if not os.path.isabs(path):
                path = os.path.join(data_dir, path)
            if not os.path.exists(path):
                logger.warning("Missing local dataset: %s", path)
                return []
            rows = _load_xlsx(path, config)
        elif source.kind == "jsonl":
            path = source.path
            if not os.path.isabs(path):
                path = os.path.join(data_dir, path)
            if not os.path.exists(path):
                logger.warning("Missing local dataset: %s", path)
                return []
            rows = _load_jsonl(path)
        else:
            from datasets import load_dataset

            # A "{config}" placeholder means one repo per subset rather than
            # one repo with several configs (SAKURA's English originals).
            templated = "{config}" in source.path
            repo = source.path.format(config=config) if templated else source.path
            try:
                dataset = load_dataset(
                    repo, name=None if templated else config, split=split,
                    token=_hf_token(source.public),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load %s [%s/%s]: %s",
                               repo, config, split, exc)
                continue

            image_column = _pick_column(dataset, source.image_columns, "image",
                                        source, config, split, logger)
            audio_column = _pick_column(dataset, source.audio_columns, "audio",
                                        source, config, split, logger)

            if audio_column is not None:
                # Take the clips as raw file bytes and decode them ourselves.
                # datasets>=5 hands decoded audio back as a torchcodec object,
                # which would make torchcodec a hard dependency of the whole
                # toolkit; soundfile reads the same bytes without it.
                try:
                    from datasets import Audio

                    dataset = dataset.cast_column(audio_column,
                                                  Audio(decode=False))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s: could not take audio undecoded (%s); "
                                   "falling back to the column as stored",
                                   source.path, exc)

            # Keep pixels and samples out of the row dicts: iterate a view
            # without them and attach a lazy loader to each Sample instead.
            heavy = [c for c in (image_column, audio_column) if c is not None]
            rows = dataset.remove_columns(heavy) if heavy else dataset

        for local_index, row in enumerate(rows):
            payload = dict(row)
            # Adapters need to know which subset they are reading; SAKURA's
            # category and SpokenMQA's difficulty are carried by the config.
            payload["__config__"] = config
            payload["__split__"] = split

            produced = source.adapter(payload, global_index)
            global_index += 1
            if produced is None:
                continue
            # One row may yield several items (SAKURA: single-hop + multi-hop).
            batch = produced if isinstance(produced, list) else [produced]

            for sample in batch:
                if sample is None:
                    continue
                # A config name is often the only carrier of the subject/subset.
                if sample.group is None and config is not None:
                    sample.group = config
                if source.group_map and sample.group in source.group_map:
                    sample.group = source.group_map[sample.group]
                if source.uid_counter:
                    base = sample.uid or sample.group or str(config)
                    sample.uid = f"{base}#{counters[base]}"
                    counters[base] += 1
                if image_column is not None:
                    sample.image_loader = _media_thunk(dataset, local_index,
                                                       image_column)
                if audio_column is not None:
                    sample.audio_loader = _media_thunk(dataset, local_index,
                                                       audio_column)
                elif manifest_rows is not None:
                    sample.audio_loader = _path_thunk(payload["__audio_path__"])
                samples.append(sample)

    samples = _ensure_unique_uids(samples, source, logger)

    if not samples and source.fallback is not None and not suppress_fallback:
        logger.warning("No local corpus configured; falling back to %s on the "
                       "Hub, which downloads the whole dataset. Set "
                       "ASR_DATA_DIR to evaluate on your own audio instead.",
                       source.fallback.path)
        return load_source(source.fallback, data_dirs, logger)
    return samples


def subsample(samples: List[Sample], portion: float, seed: int) -> List[Sample]:
    """Keep a deterministic ``portion`` of the items, selected by uid.

    Selecting on uid (rather than on position) means that two languages sharing
    a uid space end up with the same items even when sampled independently.
    """
    if portion >= 1.0 or not samples:
        return samples
    uids = sorted({s.uid for s in samples}, key=_natural_key)
    keep_n = max(1, int(len(uids) * portion))
    keep = set(random.Random(seed).sample(uids, keep_n))
    return [s for s in samples if s.uid in keep]


def load_benchmark(spec: BenchmarkSpec, languages: List[str],
                   data_dirs: Dict[str, str], portion: float, seed: int,
                   logger: logging.Logger) -> Dict[str, List[Sample]]:
    """Load every requested language of a benchmark, aligned and subsampled."""
    wanted = [lang for lang in languages if lang in spec.sources]
    loaded: Dict[str, List[Sample]] = {}
    for lang in wanted:
        samples = load_source(spec.sources[lang], data_dirs, logger)
        if samples:
            loaded[lang] = samples
        else:
            logger.warning("%s/%s: no samples loaded", spec.key, lang)

    if spec.align and len(loaded) > 1:
        shared = set.intersection(*({s.uid for s in items}
                                    for items in loaded.values()))
        for lang, items in loaded.items():
            dropped = len(items) - sum(1 for s in items if s.uid in shared)
            if dropped:
                logger.info("%s/%s: dropped %d items with no counterpart in "
                            "the other languages", spec.key, lang, dropped)
            loaded[lang] = [s for s in items if s.uid in shared]

    return {lang: subsample(items, portion, seed)
            for lang, items in loaded.items()}
