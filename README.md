# KazakhEvalKit

A multi-modal evaluation harness for LLMs served through vLLM, built around
Kazakh and its English and Russian counterparts.

39 benchmarks across five passes — text, vision, audio, safety and
translation — each reported per language, with Kazakh first.

**Every benchmark is zero-shot.** No in-context examples are ever prepended:
each item is presented on its own, with an instruction describing the answer
format and nothing else. Nothing here is few-shot or chain-of-thought prompted,
so a reasoning model's trace is its own doing rather than something the harness
asked for.

---

## Table of contents

1. [What it evaluates](#1-what-it-evaluates)
2. [How it works](#2-how-it-works)
3. [Installation](#3-installation)
4. [Credentials](#4-credentials)
5. [Serving a model](#5-serving-a-model)
6. [Running an evaluation](#6-running-an-evaluation)
7. [Working in tmux](#7-working-in-tmux)
8. [Reading the results](#8-reading-the-results)
9. [Configuration reference](#9-configuration-reference)
10. [Recipes](#10-recipes)
11. [Translation and XCOMET](#11-translation-and-xcomet)
12. [Resuming and re-judging](#12-resuming-and-re-judging)
13. [Repository layout](#13-repository-layout)

---

## 1. What it evaluates

Run `./run_eval.sh --list_benchmarks` to see this with language coverage.

### Text — 19 benchmarks

| Benchmark | Languages | Task |
|---|---|---|
| `mmlu` | kk, en | Multiple choice |
| `mmlu_pro` | kk, ru, en | Multiple choice, 10 options |
| `mmlu_redux` | kk, en | Multiple choice, error-corrected MMLU |
| `gpqa` | kk, ru, en | Graduate-level science |
| `arc` | kk, ru, en | ARC Easy + Challenge |
| `gsm8k` | kk, ru, en | Grade-school math |
| `polymath` | kk, en | Competition math, graded symbolically |
| `kazculture` | kk | Kazakh cultural knowledge |
| `kazmmlu` | kk | Kazakh school and professional exams |
| `kazbench_kk` | kk | Socio-cultural multiple choice |
| `belebele` | kk | Reading comprehension |
| `piqa` | kk | Physical commonsense |
| `include` | kk, ru | Regional exams |
| `kkcopa` | kk | Causal reasoning (bundled) |
| `nis_math` | kk | Olympiad-style math (bundled) |
| `kkwikispell` | kk | Sentence correction, exact match (bundled) |
| `kazqad` | kk | Extractive QA — **LLM judge** |
| `ragbench` | kk | Grounded generation — **LLM judge** |
| `ifbench` | kk | Instruction following — **LLM judge** |

### Vision — 8 benchmarks

| Benchmark | Languages | Task |
|---|---|---|
| `realworldqa` | kk, ru, en | Spatial reasoning |
| `mmstar` | kk, ru, en | Multimodal multiple choice |
| `mathvista` | kk, en | Visual math |
| `mathvision` | kk, en | Competition visual math |
| `mmbench` | kk, en | Multimodal multiple choice |
| `ai2d` | kk, en | Diagram understanding |
| `ocrbench` | kk, en | Text recognition — official rule (en), **judge** (kk) |
| `babyvision` | kk, en | Perception puzzles — exact + **judge** |

### Audio — 5 benchmarks

| Benchmark | Languages | Task |
|---|---|---|
| `sakura` | kk, ru, en | Audio MCQ, single- and multi-hop, 4 attributes |
| `spokenmqa` | kk, en | Spoken math, digit and reasoning subsets |
| `asr` | kk, ru, en | Speech recognition — **WER**, lower is better |
| `wavcaps` | kk, ru, en | Audio captioning — **LLM judge** |
| `wavcaps_qa` | kk, ru, en | Audio question answering — **LLM judge** |

### Safety — Qorgau

`qorgau`, Kazakh and Russian, across six risk areas. The model answers each
adversarial question and a judge grades the reply against Qorgau's own
per-risk-area rubric. The metric is the share of responses judged safe.

### Translation — FLORES

Six directions between Kazakh, Russian and English: `flores_kk_en`,
`flores_kk_ru`, `flores_ru_kk`, `flores_ru_en`, `flores_en_kk`,
`flores_en_ru`. Scored with BLEU and chrF++, and optionally XCOMET
(see [section 11](#11-translation-and-xcomet)).

### Sample sizes

Items per benchmark and language at `DATA_PORTION=1.0`. A dash means the
benchmark has no set for that language. Where a benchmark is aligned across
languages the counts match by construction; where they differ, the sets are
genuinely different sizes.

**Text — 143,460 items**

| Benchmark | Kazakh | English | Russian |
|---|---:|---:|---:|
| `mmlu` | 14,042 | 14,042 | – |
| `mmlu_pro` | 12,022 | 12,022 | 12,022 |
| `mmlu_redux` | 5,700 | 5,700 | – |
| `gpqa` | 1,192 | 1,192 | 1,192 |
| `arc` | 3,548 | 3,548 | 3,548 |
| `gsm8k` | 1,319 | 1,319 | 1,319 |
| `polymath` | 500 | 500 | – |
| `kazculture` | 1,334 | – | – |
| `kazmmlu` | 22,889 | – | – |
| `kazbench_kk` | 7,111 | – | – |
| `belebele` | 900 | – | – |
| `piqa` | 100 | – | – |
| `include` | 500 | – | 552 |
| `kkcopa` | 500 | – | – |
| `nis_math` | 100 | – | – |
| `kkwikispell` | 160 | – | – |
| `kazqad` | 2,713 | – | – |
| `ragbench` | 11,430 | – | – |
| `ifbench` | 444 | – | – |

**Vision — 31,974 items**

| Benchmark | Kazakh | English | Russian |
|---|---:|---:|---:|
| `realworldqa` | 765 | 765 | 765 |
| `mmstar` | 1,500 | 1,500 | 1,500 |
| `ai2d` | 3,088 | 3,088 | – |
| `mathvista` | 1,000 | 1,000 | – |
| `mathvision` | 3,040 | 3,040 | – |
| `mmbench` | 4,329 | 4,377 | – |
| `ocrbench` | 441 | 1,000 | – |
| `babyvision` | 388 | 388 | – |

**Audio — 26,011 items**

| Benchmark | Kazakh | English | Russian |
|---|---:|---:|---:|
| `sakura` | 4,000 | 4,000 | 4,000 |
| `spokenmqa` | 2,256 | 2,256 | – |
| `asr` | 1,225 | 1,041 | 1,131 |
| `wavcaps` | 1,730 | 1,730 | 1,730 |
| `wavcaps_qa` | 304 | 304 | 304 |

**Safety — 8,168 items**

| Benchmark | Kazakh | Russian |
|---|---:|---:|
| `qorgau` | 3,786 | 4,382 |

**Translation — 6,072 items**

| Direction | Items |
|---|---:|
| `flores_kk_en`, `flores_kk_ru` | 1,012 each |
| `flores_ru_kk`, `flores_ru_en` | 1,012 each |
| `flores_en_kk`, `flores_en_ru` | 1,012 each |

**Total: 215,685 generations for a complete run**, plus roughly 26,000 judge
calls. Use `DATA_PORTION` to work on a deterministic sample while setting up —
`0.005` gives about 1,100 items across everything.

---

## 2. How it works

```
serve_model.sh ──► vLLM OpenAI-compatible server ──► run_eval.sh
                                                          │
                                    ┌─────────────────────┼──────────────────┐
                                    ▼                     ▼                  ▼
                              generate              score locally      LLM judge
                          (concurrent HTTP)      (letters, sympy,     (OpenAI Batch
                                                   WER, BLEU)          or chat API)
                                    └─────────────────────┴──────────────────┘
                                                          ▼
                                          results/<model>/performance_report.md
```

Three things are worth knowing up front.

**Ordering.** Within each pass, all Kazakh benchmarks run first, then English,
then Russian. Everything scored locally finishes before anything that waits on
the judge API, so the GPU is never idle waiting on a queue.

**Most scoring is local.** Letters, numbers, symbolic math, the official
OCRBench rule, WER, BLEU and chrF++ all run on your machine. Only free-form
tasks with no mechanical answer reach the LLM judge — 8 of 39 benchmarks.

**Extraction failures are reported, not hidden.** The headline metric is
`accuracy_valid_only`: the score over items an answer could actually be read
from. The all-items figure sits beside it, and `extraction_failed` says how
many were dropped. If that count is high, the score rests on a small
denominator and you should look at why.

---

## 3. Installation

Requires **Python ≥ 3.12** and a machine with GPUs for the served model.

### 3.1 The evaluation environment

```bash
git clone https://github.com/akylbekmaxutov/KazakhEvalKit.git
cd KazakhEvalKit

uv venv qolda-avl-evals --python 3.12
source qolda-avl-evals/bin/activate

uv pip install -r requirements.txt
```

### 3.2 vLLM

vLLM is **not** in `requirements.txt`: its build is tied to your CUDA driver
and GPU, so it is installed separately into the same environment.

For Qolda-AVL models, install the IS2AI fork:

```bash
# Install this fork (precompiled binaries)
git clone https://github.com/IS2AI/vLLM-Qolda-AVL.git
cd vLLM-Qolda-AVL
VLLM_USE_PRECOMPILED=1 uv pip install -e .
```

For any other model, upstream vLLM is fine:

```bash
uv pip install vllm
```

### 3.3 Audio support

Only needed if you will evaluate the audio modality. vLLM decodes audio
server-side and checks for these **at start-up**, so install them *before*
launching the server:

```bash
uv pip install av resampy scipy soundfile librosa
```

### 3.4 FLEURS, for the ASR benchmark

The `asr` benchmark is scored against **FLEURS**. Download the Kazakh, Russian
and English test sets before running it — every other benchmark is fetched
automatically, this one is not.

The source is [`google/fleurs`](https://huggingface.co/datasets/google/fleurs),
configs `kk_kz`, `ru_ru` and `en_us`, `test` split.

Then give `run_eval.sh` the directory it was downloaded to:

```bash
MODALITY=audio BENCHMARKS=asr \
ASR_DATA_DIR=/path/to/fleurs_eval \
MODEL_NAME=qolda-avl-5b ./run_eval.sh
```

Or set it once as the default in `run_eval.sh`, so it need not be repeated:

```bash
ASR_DATA_DIR="${ASR_DATA_DIR:-/path/to/fleurs_eval}"
```

The directory is read as exported — nothing needs rearranging:

```
fleurs_eval/
├── val_test_kaz_kaz.tsv     Kazakh    columns: id / audio / text
├── val_test_rus_rus.tsv     Russian
├── val_test_eng_eng.tsv     English
└── wavs/...                 the audio, referenced by the manifests
```

Only the three same-language manifests are read. Translation manifests sitting
beside them (`val_test_kaz_eng.tsv` and similar) are ignored, since speech
translation is not part of this harness.

Other layouts also work. One directory per language is recognised
(`fleurs_eval/kk_kz/test.tsv`), manifests may be `.tsv`, `.csv` or `.jsonl`
with or without a header, the audio column may be named `audio`, `file_name`,
`path` or `file`, and the text column `text`, `transcription`,
`raw_transcription` or `sentence`.

Coverage is per language: if one manifest cannot be found, only that language
is affected and the others still load.

---

## 4. Credentials

Copy the template and fill it in:

```bash
cp .env.example .env
```

```ini
# Required for the gated issai/* datasets on the Hugging Face Hub.
HF_TOKEN=hf_xxx

# JUDGE_API_KEY wins; OPENAI_API_KEY is the fallback.
JUDGE_API_KEY=sk-xxx
```

`.env` is git-ignored. It is read from the repository root regardless of your
working directory, and the run **refuses to start** if `JUDGE=true` but no key
resolves — rather than discovering it after hours of generation.

---

## 5. Serving a model

Edit the `CONFIG` block in `serve_model.sh`, or pass values on the command
line — the command line wins:

```bash
MODEL_PATH=/path/to/Qolda-AVL-5B \
SERVED_MODEL_NAME=qolda-avl-5b \
CUDA_DEVICES=0,1,2,3 \
TENSOR_PARALLEL_SIZE=4 \
REASONING_PARSER=qwen3 \
./serve_model.sh
```

| Variable | Default | Notes |
|---|---|---|
| `MODEL_PATH` | `/path_to_model` | Local path or Hub id |
| `SERVED_MODEL_NAME` | basename of `MODEL_PATH` | **Must equal `MODEL_NAME` in `run_eval.sh`** |
| `CUDA_DEVICES` | `0` | Sets `CUDA_VISIBLE_DEVICES` |
| `TENSOR_PARALLEL_SIZE` | `1` | Match the number of GPUs |
| `MAX_MODEL_LEN` | `65536` | |
| `MAX_NUM_SEQS` | `512` | Keep ≥ `CONCURRENCY` in `run_eval.sh` |
| `PORT` | `8000` | |
| `LIMIT_MM_PER_PROMPT` | `{"audio": 4, "image": 4}` | Multimodal models; empty to omit |
| `REASONING_PARSER` | `qwen3` | Returns the trace in `reasoning_content`; empty for non-reasoning models |
| `EXTRA_ARGS` | empty | Anything else passed to `vllm serve` |

`--trust-remote-code` is always on.

Wait for `Application startup complete` before starting an evaluation.

---

## 6. Running an evaluation

**Serve the model first, then run the evaluation.** `run_eval.sh` is a client:
it talks to the server over HTTP and does no model loading of its own. If the
server is not up, every request fails.

The full example, for Qolda-AVL-5B:

```bash
MODALITY=all SAFETY_EVAL=true TRANSLATION_EVAL=true \
MODEL_NAME=qolda-avl-5b MODEL_NAME_FOLDER=qolda-avl-5b \
LANGUAGES=all BENCHMARKS=all DATA_PORTION=1.0 \
JUDGE=true THINKING=auto \
CONCURRENCY=128 CONCURRENCY_VISION=32 CONCURRENCY_AUDIO=32 \
XCOMET=false \
./run_eval.sh 2>&1 | tee qolda-avl-5b.log
```

That single command runs text → vision → audio → safety → translation.

Before committing to a full run, smoke-test on half a percent of the data.
It exercises every benchmark, every language and every scoring path in
minutes:

```bash
MODALITY=all SAFETY_EVAL=true TRANSLATION_EVAL=true \
MODEL_NAME=qolda-avl-5b MODEL_NAME_FOLDER=smoke \
LANGUAGES=all BENCHMARKS=all DATA_PORTION=0.005 \
JUDGE=true CONCURRENCY=64 CONCURRENCY_VISION=8 CONCURRENCY_AUDIO=8 \
./run_eval.sh 2>&1 | tee smoke.log
```

Then check one record before trusting anything:

```bash
head -1 results/smoke/text/records/kkcopa.kk.jsonl | python -m json.tool
```

`thinking` and `response` must be cleanly separated, and `prediction` must be
a letter. If `extraction_failed` is high in `summary.json`, fix `THINKING` and
`THINK_END_TOKEN` before going further — everything downstream depends on it.

---

## 7. Working in tmux

A full run takes hours. Run both processes inside **one tmux session, in two
windows** — the server in the first, the evaluation in the second — so that a
dropped SSH connection kills neither.

```bash
tmux new -s evals                       # creates the session, window 0
```

In **window 0**, start the server:

```bash
source qolda-avl-evals/bin/activate
cd KazakhEvalKit
MODEL_PATH=/path/to/Qolda-AVL-5B SERVED_MODEL_NAME=qolda-avl-5b ./serve_model.sh
```

Wait for `Application startup complete`. Then open **window 1** with
`Ctrl-b c` and start the evaluation there:

```bash
source qolda-avl-evals/bin/activate
cd KazakhEvalKit
MODALITY=all SAFETY_EVAL=true TRANSLATION_EVAL=true \
MODEL_NAME=qolda-avl-5b MODEL_NAME_FOLDER=qolda-avl-5b \
LANGUAGES=all BENCHMARKS=all DATA_PORTION=1.0 JUDGE=true \
CONCURRENCY=128 CONCURRENCY_VISION=32 CONCURRENCY_AUDIO=32 \
./run_eval.sh 2>&1 | tee qolda-avl-5b.log
```

Useful keys: `Ctrl-b 0` / `Ctrl-b 1` switch windows, `Ctrl-b d` detaches,
`tmux attach -t evals` comes back.

**Use the same environment in both windows.** A server started from one
virtualenv and an evaluation run from another is the most common cause of
confusing failures — particularly for audio, where the server needs `av`,
`soundfile` and friends that the evaluation side does not.

---

## 8. Reading the results

Everything for one model lives in one directory:

```
results/qolda-avl-5b/
├── performance_report.md          every modality, one document
├── text/
│   ├── summary.json               scores, per-category breakdowns, settings
│   ├── records/mmlu.kk.jsonl      one line per item
│   ├── judge/                     judge requests and verdicts
│   └── eval.log
├── vision/  audio/  safety/  translation/
```

`performance_report.md` is rebuilt after every pass, so it is complete as soon
as the last one finishes and stays correct if you re-run only one modality. It
holds run settings, a benchmark × language table per modality, per-category
breakdowns for SAKURA, SpokenMQA and Qorgau, and coverage counts.

Each record is one item:

```json
{
  "uid": "...", "benchmark": "mmlu", "language": "kk",
  "reference": "B", "prediction": "B", "correct": true,
  "thinking": "...", "response": "...",
  "finish_reason": "stop", "truncated_thinking": false
}
```

`thinking` and `response` are kept when `SAVE_RESPONSES=true` (the default).

### How to read the metrics

| Metric | Scale | Meaning |
|---|---|---|
| `accuracy_valid_only` | percent | **The headline.** Over items an answer could be read from |
| `accuracy` | percent | Over all items, including unreadable ones |
| `wer` | 0–1, four decimals | Word error rate, **lower is better** |
| `xcomet`, `bleu`, `chrf2` | 0–1, four decimals | Translation quality |
| `extraction_failed` | count | No parseable answer — a format problem, not a wrong answer |
| `unscorable_samples` | count | Could not be graded at all (judge missing, symbolic check undecided) |

---

## 9. Configuration reference

Every variable below can be set on the command line or edited into
`run_eval.sh`. The command line wins, so the file holds your defaults and the
command line is the per-run override:

```bash
# on the command line
CONCURRENCY=128 TEMPERATURE=0.6 ./run_eval.sh

# or edited into run_eval.sh
CONCURRENCY="${CONCURRENCY:-128}"
TEMPERATURE="${TEMPERATURE:-0.6}"
```

### 9.1 What to run

| Variable | Default | Meaning |
|---|---|---|
| `MODALITY` | `text` | `text`, `vision`, `audio`, `all`, `none`, or a comma-separated list such as `text,vision` |
| `SAFETY_EVAL` | `false` | Also run the Qorgau safety pass |
| `TRANSLATION_EVAL` | `false` | Also run the FLORES translation pass |
| `BENCHMARKS` | `all` | Comma-separated benchmark keys |
| `LANGUAGES` | `all` | Any subset of `kk,ru,en` |

### 9.2 Connecting to the served model

| Variable | Default | Meaning |
|---|---|---|
| `API_BASE` | `http://localhost:8000/v1` | **Set this when the server is on another port or host** |
| `API_KEY` | `EMPTY` | Only if vLLM was started with `--api-key` |
| `MODEL_NAME` | `qolda-avl` | Must equal `--served-model-name` on the server |
| `MODEL_NAME_FOLDER` | `$MODEL_NAME` | Results directory name; use it to separate runs of the same model |
| `OUTPUT_ROOT` | `results` | Where result folders are created |
| `SYSTEM_PROMPT` | empty | Prepended to every request |
| `CONCURRENCY` | `32` | Requests in flight; also bounds media decoding |
| `CONCURRENCY_VISION` | `$CONCURRENCY` | Vision pass only — images are large, 8–32 is sensible |
| `CONCURRENCY_AUDIO` | `$CONCURRENCY` | Audio pass only |
| `RETRIES` | `5` | Retries per failed request |
| `REQUEST_TIMEOUT` | `1800` | Seconds per request |

Serving on a non-default port:

```bash
API_BASE=http://localhost:8003/v1 MODEL_NAME=qolda-think ./run_eval.sh
```

A server on another machine:

```bash
API_BASE=http://gpu-node-04:8000/v1 MODEL_NAME=qolda-think ./run_eval.sh
```

### 9.3 Sampling

| Variable | Default | Meaning |
|---|---|---|
| `TEMPERATURE` | `1.0` | |
| `TOP_P` | `0.95` | |
| `TOP_K` | `20` | vLLM extension, sent in `extra_body` |
| `MIN_P` | `0.0` | vLLM extension |
| `PRESENCE_PENALTY` | `1.5` | Vision overrides this to `0.0` |
| `REPETITION_PENALTY` | `1.0` | vLLM extension |
| `MAX_TOKENS` | `32768` | Generation budget. With `THINKING=on` it covers the reasoning trace too |
| `SAMPLING_SEED` | empty | Server-side seed; empty is non-deterministic |

A model card's recommended settings, passed straight through:

```bash
TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20 MIN_P=0 \
PRESENCE_PENALTY=1.5 REPETITION_PENALTY=1.0 MAX_TOKENS=16384 \
MODEL_NAME=qolda-think ./run_eval.sh
```

`PRESENCE_PENALTY` is the one non-neutral default here — if your model card
specifies one, pass it explicitly rather than inheriting `1.5`.

### 9.4 Per-pass overrides

One model often wants different decoding per modality. These take **any**
`run_eval` flag, not only sampling ones, and apply to that pass alone:

| Variable | Default | Applies to |
|---|---|---|
| `TEXT_SAMPLING` | empty | text |
| `VISION_SAMPLING` | `--presence_penalty 0.0` | vision |
| `AUDIO_SAMPLING` | empty | audio |
| `SAFETY_SAMPLING` | empty | safety |
| `TRANSLATION_SAMPLING` | empty | translation |

```bash
MODALITY=all \
TEXT_SAMPLING="--max_tokens 32768" \
VISION_SAMPLING="--temperature 0.2 --presence_penalty 0.0 --max_tokens 4096" \
AUDIO_SAMPLING="--temperature 0.4 --max_tokens 2048" \
MODEL_NAME=qolda-avl-5b ./run_eval.sh
```

### 9.5 Reasoning

| Variable | Default | Meaning |
|---|---|---|
| `THINKING` | `auto` | `auto` leaves the chat template alone — correct for reasoning-native models. `on`/`off` set `enable_thinking` |
| `THINK_START_TOKEN` | `<think>` | Used when the trace is inline rather than in `reasoning_content` |
| `THINK_END_TOKEN` | `</think>` | |
| `SAVE_RESPONSES` | `true` | Keep the trace and the reply in every record |
| `STRUCTURED_OUTPUT` | `false` | Guided decoding. **Leave off for reasoning models**: it forces the first token to be `{`, so `<think>` can never open |

### 9.6 Data

| Variable | Default | Meaning |
|---|---|---|
| `DATA_PORTION` | `1.0` | `0.05` takes a deterministic 5% sample |
| `SEED` | `42` | Subsampling seed, not decoding |
| `DATA_DIR` | `datasets` | Bundled local benchmarks |
| `RESUME` | `false` | Reuse records already on disk |
| `RETRY_UNPARSED` | `1` | Extra passes over items with no parseable answer; `0` disables |
| `REJUDGE` | `false` | Discard existing judge verdicts and grade again |
| `PROMPT_LANG` | `en` | `en` keeps harness instructions English; `auto` matches the benchmark |
| `ASR_DATA_DIR` | `/path/to/fleurs_eval` | Your downloaded FLEURS corpus — see [section 3.4](#34-fleurs-for-the-asr-benchmark) |

### 9.7 Vision

| Variable | Default | Meaning |
|---|---|---|
| `IMAGE_MAX_SIZE` | `1024` | Longest edge in pixels; `0` keeps the original. Keep it capped — dynamic-resolution VLMs turn pixels into vision tokens with no upper bound |
| `IMAGE_FORMAT` | `JPEG` | `JPEG`, `PNG` or `WEBP` |
| `IMAGE_QUALITY` | `90` | JPEG/WEBP quality |
| `NO_SYSTEM_ROLE` | auto | Gemma-3, InternVL and DeepSeek-VL reject a system turn beside an image; auto-detected from `MODEL_NAME` |

### 9.8 Audio

| Variable | Default | Meaning |
|---|---|---|
| `AUDIO_SAMPLING_RATE` | `16000` | Every clip is resampled to this on decode |
| `AUDIO_FORMAT` | `WAV` | `WAV`, `FLAC` or `OGG` |
| `AUDIO_CONTENT_TYPE` | `audio_url` | What vLLM accepts; `input_audio` is the OpenAI schema |
| `AUDIO_PLACEHOLDER` | empty | Set `<audio>` if your chat template does not insert its own marker |
| `AUDIO_MAX_SECONDS` | `0` | Truncate longer clips; `0` keeps them whole |

### 9.9 Translation metric

| Variable | Default | Meaning |
|---|---|---|
| `XCOMET` | `false` | `true` scores with XCOMET; `false` saves translations and reports BLEU/chrF++ only |
| `XCOMET_MODEL` | `Unbabel/XCOMET-XXL` | `Unbabel/XCOMET-XL` is lighter |
| `XCOMET_BATCH_SIZE` | `8` | |
| `XCOMET_GPUS` | `1` | |
| `XCOMET_CACHE_DIR` | empty | comet's own cache when empty |

### 9.10 The judge

| Variable | Default | Meaning |
|---|---|---|
| `JUDGE` | `true` | Run the judge where a benchmark needs it |
| `JUDGE_MODEL` | `gpt-5.6-luna` | |
| `JUDGE_REASONING_EFFORT` | `none` | |
| `JUDGE_MAX_COMPLETION_TOKENS` | `32` | **Covers reasoning tokens too.** Effort `medium` needs ~512, or the reply comes back empty |
| `JUDGE_STRUCTURED` | `true` | Strict JSON schema, so a verdict cannot come back malformed |
| `JUDGE_DIRECT_BELOW` | `500` | Smaller sets skip the Batch API |
| `JUDGE_DIRECT_CONCURRENCY` | `1` | Judge requests in flight in total; `1` is strictly sequential |
| `JUDGE_DIRECT_MAX_ATTEMPTS` | `2` | Tries per direct request, with backoff |
| `JUDGE_SPLIT_ABOVE` | `5000` | Larger sets split into several batch jobs |
| `JUDGE_SPLIT_PARTS` | `3` | How many |
| `JUDGE_MAX_PARALLEL` | `4` | Batch jobs in flight |
| `JUDGE_COMPLETION_WINDOW` | `24h` | Batch API completion window |
| `JUDGE_POLL_INTERVAL` | `30` | Seconds between batch status checks |
| `JUDGE_TIMEOUT` | `86400` | Seconds to wait for a batch before giving up |

### 9.11 Flags with no variable

A few options have no `run_eval.sh` variable. Pass them as arguments instead —
anything after `./run_eval.sh` is forwarded to every pass:

```bash
./run_eval.sh --list_benchmarks        # what is available, then exit
./run_eval.sh --verbose                # per-item warnings on stderr
./run_eval.sh --image_detail low       # OpenAI image detail hint
./run_eval.sh --judge_base_url https://my-proxy/v1
```

`--judge_api_key` also exists, but prefer `JUDGE_API_KEY` in `.env` so the key
never reaches your shell history.

## 10. Recipes

**Everything** — all five passes, all benchmarks, all languages, full data.
This is the complete evaluation:

```bash
MODALITY=all SAFETY_EVAL=true TRANSLATION_EVAL=true \
MODEL_NAME=qolda-avl-5b MODEL_NAME_FOLDER=qolda-avl-5b \
LANGUAGES=all BENCHMARKS=all DATA_PORTION=1.0 \
JUDGE=true THINKING=auto \
CONCURRENCY=128 CONCURRENCY_VISION=32 CONCURRENCY_AUDIO=32 \
XCOMET=false \
./run_eval.sh 2>&1 | tee qolda-avl-5b.log
```

**Everything except audio** — for a model with no audio encoder:

```bash
MODALITY=text,vision SAFETY_EVAL=true TRANSLATION_EVAL=true \
MODEL_NAME=my-model MODEL_NAME_FOLDER=my-model \
LANGUAGES=all BENCHMARKS=all DATA_PORTION=1.0 JUDGE=true \
CONCURRENCY=128 CONCURRENCY_VISION=32 ./run_eval.sh
```

**A single benchmark:**

```bash
MODALITY=vision BENCHMARKS=mmstar LANGUAGES=kk \
MODEL_NAME=qolda-avl-5b ./run_eval.sh
```

**Kazakh only, across everything:**

```bash
MODALITY=all LANGUAGES=kk MODEL_NAME=qolda-avl-5b ./run_eval.sh
```

**Safety on its own:**

```bash
MODALITY=none SAFETY_EVAL=true LANGUAGES=kk,ru \
MODEL_NAME=qolda-avl-5b ./run_eval.sh
```

**Generation now, judging later** — keeps the GPU busy and defers every API
queue wait to the end:

```bash
MODALITY=all JUDGE=false MODEL_NAME=qolda-avl-5b ./run_eval.sh   # GPU phase
MODALITY=all JUDGE=true RESUME=true MODEL_NAME=qolda-avl-5b ./run_eval.sh
```

**Compare thinking on and off** — same served model, two result folders. Use a
model whose chat template supports the switch; a reasoning-only model ignores
`THINKING=off` and should be left on `auto`:

```bash
THINKING=on  MODEL_NAME=qolda MODEL_NAME_FOLDER=qolda-think   ./run_eval.sh
THINKING=off MODEL_NAME=qolda MODEL_NAME_FOLDER=qolda-nothink ./run_eval.sh
```

---

## 11. Translation and XCOMET

The translation pass always reports **BLEU** and **chrF++**, computed locally.

**XCOMET is optional and needs its own environment.** `unbabel-comet`
requires `huggingface-hub<1.0`, `transformers<5.0` and `numpy<2.0`, while
current `transformers` requires `huggingface-hub>=1.5`. Installing it beside
`requirements.txt` downgrades numpy and transformers and breaks `datasets`.
So the two never share an environment.

Step 1 — generate translations in the main environment (`XCOMET=false` is the
default):

```bash
MODALITY=none TRANSLATION_EVAL=true XCOMET=false \
MODEL_NAME=qolda-avl-5b ./run_eval.sh
```

Step 2 — score them in a separate environment. `RESUME=true` reads the
translations off disk, so nothing is regenerated and the server does not need
to be running:

```bash
uv venv xcomet-env --python 3.12 && source xcomet-env/bin/activate
uv pip install -r requirements-xcomet.txt

MODALITY=none TRANSLATION_EVAL=true XCOMET=true RESUME=true \
MODEL_NAME=qolda-avl-5b XCOMET_GPUS=1 ./run_eval.sh
```

XCOMET-XXL is 10.7B and gated: accept the licence at
[Unbabel/XCOMET-XXL](https://huggingface.co/Unbabel/XCOMET-XXL) with the
account behind `HF_TOKEN`, and give it a free GPU.
`XCOMET_MODEL=Unbabel/XCOMET-XL` (3.5B) is a lighter option.

---

## 12. Resuming and re-judging

`RESUME=true` reuses every record already on disk and generates only what is
missing. Use the **same `DATA_PORTION` and `SEED`** as the original run — a
different portion draws a different sample and nothing matches.

```bash
MODALITY=all RESUME=true MODEL_NAME=qolda-avl-5b ./run_eval.sh
```

A resume is safe to repeat. Items that produced no parseable answer are
retried once (`RETRY_UNPARSED`), and that budget is stored per item, so a
second resume does not spend it again. Wrong answers are never regenerated —
only empty ones — so retrying cannot inflate a score.

`REJUDGE=true` discards verdicts a previous judge run produced and grades them
again, reusing all generation. Use it after changing the judge model, its
effort or its prompt; without it a resumed run sees every judged item already
settled and does nothing.

```bash
MODALITY=vision BENCHMARKS=babyvision RESUME=true REJUDGE=true \
MODEL_NAME=qolda-avl-5b ./run_eval.sh
```

---

## 13. Repository layout

```
KazakhEvalKit/
├── run_eval.sh              one runner for every pass
├── serve_model.sh           vLLM launcher
├── requirements.txt         evaluation dependencies
├── requirements-xcomet.txt  XCOMET, for a separate environment
├── .env.example             credential template
├── core/                    modality-agnostic engine
│   ├── runner.py            load → generate → score → judge → report
│   ├── loaders.py           Hub, JSONL and manifest loading, alignment
│   ├── generation.py        the vLLM client
│   ├── extraction.py        reasoning/answer separation
│   ├── scoring/             symbolic.py, rules.py, judge.py
│   ├── reporting.py         records, metric blocks, summaries
│   └── report_md.py         performance_report.md
├── text_modality/           specs, prompts, scoring per modality
├── vision_modality/
├── audio_modality/
├── safety/
├── translation/
└── datasets/                bundled benchmarks (kkCOPA, NIS Math, …)
```

Each modality package holds `specs.py` (which benchmark, which source, which
language), `prompts.py` and `modality.py` (extraction and scoring). Adding a
benchmark means adding a `BenchmarkSpec` and an adapter function — the runner,
the judge and the reporting need no changes.
