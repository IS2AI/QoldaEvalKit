#!/usr/bin/env bash
# KazakhEvalKit — one runner for every modality. See README.md.
#
#   MODALITY=all ./run_eval.sh
#   MODALITY=text,vision ./run_eval.sh                 # everything but audio
#   SAFETY_EVAL=true MODALITY=none ./run_eval.sh       # safety only
#   MODALITY=vision BENCHMARKS=ai2d,mmstar ./run_eval.sh
#   ./run_eval.sh --list_benchmarks
#
# Results: $OUTPUT_ROOT/$MODEL_NAME_FOLDER/{performance_report.md,<modality>/}
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# Keep Python line-buffered so progress is visible through `tee`.
export PYTHONUNBUFFERED=1

# =============================================================================
# 1. WHAT TO RUN
# =============================================================================

# text | vision | audio | all | none | comma-separated list
MODALITY="${MODALITY:-text}"

SAFETY_EVAL="${SAFETY_EVAL:-false}"          # Qorgau kk/ru safety pass
TRANSLATION_EVAL="${TRANSLATION_EVAL:-false}"  # FLORES translation pass

# "all", or a comma-separated list of benchmark keys.
#   text   : mmlu mmlu_pro gpqa arc gsm8k mmlu_redux polymath kazculture
#            kazmmlu kazbench_kk belebele piqa include kkcopa nis_math
#            kkwikispell kazqad ragbench ifbench
#   vision : realworldqa mmstar ai2d mathvista mathvision mmbench ocrbench
#            babyvision
#   audio  : sakura spokenmqa wavcaps wavcaps_qa asr
BENCHMARKS="${BENCHMARKS:-all}"

# "all", or any comma-separated subset of kk,ru,en.
LANGUAGES="${LANGUAGES:-all}"

# =============================================================================
# 2. THE SERVED MODEL (must match how serve_model.sh was launched)
# =============================================================================

API_BASE="${API_BASE:-http://localhost:8000/v1}"
API_KEY="${API_KEY:-EMPTY}"

MODEL_NAME="${MODEL_NAME:-qolda-avl}"        # = --served-model-name
MODEL_NAME_FOLDER="${MODEL_NAME_FOLDER:-$MODEL_NAME}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results}"

SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"

# Requests in flight against vLLM. Bounds media decoding too, so at most this
# many images/clips are held in memory at once.
CONCURRENCY="${CONCURRENCY:-32}"
CONCURRENCY_VISION="${CONCURRENCY_VISION:-$CONCURRENCY}"
CONCURRENCY_AUDIO="${CONCURRENCY_AUDIO:-$CONCURRENCY}"
RETRIES="${RETRIES:-5}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1800}"

# =============================================================================
# 3. SAMPLING
# =============================================================================

TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0.0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
SAMPLING_SEED="${SAMPLING_SEED:-}"           # empty = non-deterministic

# Per-pass overrides. Any run_eval flag is accepted, not just sampling ones:
#   VISION_SAMPLING="--temperature 0.2 --top_p 0.9 --max_tokens 4096"
TEXT_SAMPLING="${TEXT_SAMPLING:-}"
VISION_SAMPLING="${VISION_SAMPLING:---presence_penalty 0.0}"
AUDIO_SAMPLING="${AUDIO_SAMPLING:-}"
SAFETY_SAMPLING="${SAFETY_SAMPLING:-}"
TRANSLATION_SAMPLING="${TRANSLATION_SAMPLING:-}"

# =============================================================================
# 4. REASONING / THINKING
# =============================================================================

THINKING="${THINKING:-auto}"                 # auto | on | off
THINK_START_TOKEN="${THINK_START_TOKEN:-<think>}"
THINK_END_TOKEN="${THINK_END_TOKEN:-</think>}"
SAVE_RESPONSES="${SAVE_RESPONSES:-true}"

# Guided decoding for the evaluated model. Leave false for reasoning models:
# it forces the first token to be "{", so <think> can never open.
STRUCTURED_OUTPUT="${STRUCTURED_OUTPUT:-false}"

# =============================================================================
# 5. DATA
# =============================================================================

DATA_PORTION="${DATA_PORTION:-1.0}"          # 0.1 = a deterministic 10% sample
SEED="${SEED:-42}"                           # subsampling seed, not decoding
DATA_DIR="${DATA_DIR:-datasets}"
RESUME="${RESUME:-false}"                    # reuse records already on disk

# Extra passes over items no answer could be parsed from. Only empty
# predictions are retried, never wrong ones, so it cannot inflate a score.
RETRY_UNPARSED="${RETRY_UNPARSED:-1}"

# Discard existing judge verdicts and grade again. Needed after changing the
# judge model, effort or prompt; generation is still reused.
REJUDGE="${REJUDGE:-false}"

# en   : harness instructions in English, questions stay in their own language
# auto : instructions match the benchmark's language
PROMPT_LANG="${PROMPT_LANG:-en}"

# Your downloaded FLEURS corpus (kk, ru, en). See README section 3.4.
ASR_DATA_DIR="${ASR_DATA_DIR:-/path/to/fleurs_eval}"

# =============================================================================
# 6. MEDIA
# =============================================================================

IMAGE_FORMAT="${IMAGE_FORMAT:-JPEG}"
# Longest edge in pixels; 0 keeps the original. Keep this capped: dynamic-
# resolution VLMs turn pixels straight into vision tokens with no upper bound.
IMAGE_MAX_SIZE="${IMAGE_MAX_SIZE:-1024}"
IMAGE_QUALITY="${IMAGE_QUALITY:-90}"
# Gemma-3 / InternVL / DeepSeek-VL reject a system turn beside an image.
# Empty auto-detects from MODEL_NAME.
NO_SYSTEM_ROLE="${NO_SYSTEM_ROLE:-}"

AUDIO_SAMPLING_RATE="${AUDIO_SAMPLING_RATE:-16000}"
AUDIO_FORMAT="${AUDIO_FORMAT:-WAV}"
AUDIO_CONTENT_TYPE="${AUDIO_CONTENT_TYPE:-audio_url}"   # or input_audio
AUDIO_PLACEHOLDER="${AUDIO_PLACEHOLDER:-}"   # set '<audio>' if the template needs it
AUDIO_MAX_SECONDS="${AUDIO_MAX_SECONDS:-0}"  # 0 keeps clips whole

# =============================================================================
# 7. TRANSLATION METRIC (XCOMET, local GPU, separate environment)
# =============================================================================

XCOMET_MODEL="${XCOMET_MODEL:-Unbabel/XCOMET-XXL}"
XCOMET_BATCH_SIZE="${XCOMET_BATCH_SIZE:-8}"
XCOMET_GPUS="${XCOMET_GPUS:-1}"
XCOMET_CACHE_DIR="${XCOMET_CACHE_DIR:-}"
# false saves translations unscored and reports BLEU/chrF++ only.
XCOMET="${XCOMET:-false}"

# =============================================================================
# 8. LLM JUDGE
# =============================================================================
# Needs JUDGE_API_KEY or OPENAI_API_KEY in .env. Used by text kazqad, ragbench,
# ifbench; vision ocrbench (kk), babyvision; audio wavcaps, wavcaps_qa.

JUDGE="${JUDGE:-true}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.6-luna}"
JUDGE_REASONING_EFFORT="${JUDGE_REASONING_EFFORT:-none}"
# Covers reasoning tokens as well as the verdict; effort "medium" wants ~512.
JUDGE_MAX_COMPLETION_TOKENS="${JUDGE_MAX_COMPLETION_TOKENS:-32}"
JUDGE_COMPLETION_WINDOW="${JUDGE_COMPLETION_WINDOW:-24h}"
JUDGE_POLL_INTERVAL="${JUDGE_POLL_INTERVAL:-30}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-86400}"
JUDGE_STRUCTURED="${JUDGE_STRUCTURED:-true}"

# Sets smaller than this skip the Batch API and go through the chat API.
JUDGE_DIRECT_BELOW="${JUDGE_DIRECT_BELOW:-500}"
# Judge requests in flight, across all direct jobs. 1 = strictly sequential.
JUDGE_DIRECT_CONCURRENCY="${JUDGE_DIRECT_CONCURRENCY:-1}"
JUDGE_DIRECT_MAX_ATTEMPTS="${JUDGE_DIRECT_MAX_ATTEMPTS:-2}"
JUDGE_SPLIT_ABOVE="${JUDGE_SPLIT_ABOVE:-5000}"
JUDGE_SPLIT_PARTS="${JUDGE_SPLIT_PARTS:-3}"
JUDGE_MAX_PARALLEL="${JUDGE_MAX_PARALLEL:-4}"

# =============================================================================
# Dispatch — nothing below normally needs editing.
# =============================================================================

case "$MODALITY" in
  all)  MODALITIES=(text vision audio) ;;
  none) MODALITIES=() ;;
  *)
    IFS=',' read -r -a MODALITIES <<< "$MODALITY"
    for m in "${MODALITIES[@]}"; do
      case "$m" in
        text|vision|audio) ;;
        *) echo "MODALITY must be text, vision, audio, all, none, or a" \
                "comma-separated list of the first three (got '$m')" >&2
           exit 2 ;;
      esac
    done
    ;;
esac

case "$SAFETY_EVAL" in
  true|1|yes|on)   RUN_SAFETY=true ;;
  false|0|no|off)  RUN_SAFETY=false ;;
  *) echo "SAFETY_EVAL must be true or false (got '$SAFETY_EVAL')" >&2; exit 2 ;;
esac

case "$TRANSLATION_EVAL" in
  true|1|yes|on)   RUN_TRANSLATION=true ;;
  false|0|no|off)  RUN_TRANSLATION=false ;;
  *) echo "TRANSLATION_EVAL must be true or false (got '$TRANSLATION_EVAL')" >&2
     exit 2 ;;
esac

if [[ ${#MODALITIES[@]} -eq 0 && "$RUN_SAFETY" != "true" \
      && "$RUN_TRANSLATION" != "true" ]]; then
  echo "Nothing to run: MODALITY=none, SAFETY_EVAL=false, TRANSLATION_EVAL=false." >&2
  exit 2
fi

# With several modalities, each ignores benchmark keys it does not own.
SKIP_UNKNOWN=false
if (( ${#MODALITIES[@]} > 1 )); then
  SKIP_UNKNOWN=true
fi

if [[ "$LANGUAGES" == "all" ]]; then
  LANGUAGES="kk,ru,en"
fi

common_args=(
  --api_base            "$API_BASE"
  --api_key             "$API_KEY"
  --model_name          "$MODEL_NAME"
  --run_name            "$MODEL_NAME_FOLDER"
  --output_root         "$OUTPUT_ROOT"
  --system_prompt       "$SYSTEM_PROMPT"
  --concurrency         "$CONCURRENCY"
  --retries             "$RETRIES"
  --request_timeout     "$REQUEST_TIMEOUT"
  --benchmarks          "$BENCHMARKS"
  --skip_unknown_benchmarks "$SKIP_UNKNOWN"
  --languages           "$LANGUAGES"
  --data_portion        "$DATA_PORTION"
  --seed                "$SEED"
  --data_dir            "$DATA_DIR"
  --resume              "$RESUME"
  --rejudge             "$REJUDGE"
  --retry_unparsed      "$RETRY_UNPARSED"
  --prompt_lang         "$PROMPT_LANG"
  --temperature         "$TEMPERATURE"
  --top_p               "$TOP_P"
  --top_k               "$TOP_K"
  --min_p               "$MIN_P"
  --presence_penalty    "$PRESENCE_PENALTY"
  --repetition_penalty  "$REPETITION_PENALTY"
  --max_tokens          "$MAX_TOKENS"
  --thinking            "$THINKING"
  --think_start_token   "$THINK_START_TOKEN"
  --think_end_token     "$THINK_END_TOKEN"
  --save_responses      "$SAVE_RESPONSES"
  --structured_output   "$STRUCTURED_OUTPUT"
  --judge               "$JUDGE"
  --judge_model         "$JUDGE_MODEL"
  --judge_reasoning_effort  "$JUDGE_REASONING_EFFORT"
  --judge_max_completion_tokens "$JUDGE_MAX_COMPLETION_TOKENS"
  --judge_completion_window "$JUDGE_COMPLETION_WINDOW"
  --judge_poll_interval "$JUDGE_POLL_INTERVAL"
  --judge_timeout       "$JUDGE_TIMEOUT"
  --judge_structured    "$JUDGE_STRUCTURED"
  --judge_direct_below  "$JUDGE_DIRECT_BELOW"
  --judge_direct_concurrency  "$JUDGE_DIRECT_CONCURRENCY"
  --judge_direct_max_attempts "$JUDGE_DIRECT_MAX_ATTEMPTS"
  --judge_split_above   "$JUDGE_SPLIT_ABOVE"
  --judge_split_parts   "$JUDGE_SPLIT_PARTS"
  --judge_max_parallel  "$JUDGE_MAX_PARALLEL"
)

[[ -n "$SAMPLING_SEED" ]] && common_args+=(--sampling_seed "$SAMPLING_SEED")

# Safety and translation own their benchmark keys; a selection meant for
# another modality must not be forced on them.
SAFETY_BENCHMARKS="all"
if [[ "$BENCHMARKS" == *qorgau* ]]; then
  SAFETY_BENCHMARKS="$BENCHMARKS"
fi

TRANSLATION_BENCHMARKS="all"
if [[ "$BENCHMARKS" == *flores* ]]; then
  TRANSLATION_BENCHMARKS="$BENCHMARKS"
fi

rebuild_args() {   # $1 = benchmark selection
  pass_args=()
  for arg in "${common_args[@]}"; do pass_args+=("$arg"); done
  for i in "${!pass_args[@]}"; do
    if [[ "${pass_args[$i]}" == "--benchmarks" ]]; then
      pass_args[$((i + 1))]="$1"
    fi
    if [[ "${pass_args[$i]}" == "--skip_unknown_benchmarks" ]]; then
      pass_args[$((i + 1))]="false"
    fi
  done
}

status=0
for modality in "${MODALITIES[@]}"; do
  args=("${common_args[@]}")

  if [[ "$modality" == "vision" ]]; then
    args+=(
      --concurrency    "$CONCURRENCY_VISION"
      --image_format   "$IMAGE_FORMAT"
      --image_max_size "$IMAGE_MAX_SIZE"
      --image_quality  "$IMAGE_QUALITY"
    )
    [[ -n "$NO_SYSTEM_ROLE" ]] && args+=(--no_system_role "$NO_SYSTEM_ROLE")
  fi

  if [[ "$modality" == "audio" ]]; then
    args+=(
      --concurrency         "$CONCURRENCY_AUDIO"
      --audio_sampling_rate "$AUDIO_SAMPLING_RATE"
      --audio_format        "$AUDIO_FORMAT"
      --audio_content_type  "$AUDIO_CONTENT_TYPE"
      --audio_placeholder   "$AUDIO_PLACEHOLDER"
      --audio_max_seconds   "$AUDIO_MAX_SECONDS"
      --asr_data_dir        "$ASR_DATA_DIR"
    )
  fi

  if (( ${#MODALITIES[@]} > 1 )); then
    echo
    echo "############################  $modality  ############################"
  fi

  case "$modality" in
    text)   pass_sampling="$TEXT_SAMPLING" ;;
    vision) pass_sampling="$VISION_SAMPLING" ;;
    audio)  pass_sampling="$AUDIO_SAMPLING" ;;
    *)      pass_sampling="" ;;
  esac

  # A failing modality must not cancel the rest; the exit status still shows it.
  # shellcheck disable=SC2086
  python -m "${modality}_modality" "${args[@]}" $pass_sampling "$@" || status=$?
done

if [[ "$RUN_SAFETY" == "true" ]]; then
  echo
  echo "############################  safety  ############################"
  rebuild_args "$SAFETY_BENCHMARKS"
  # shellcheck disable=SC2086
  python -m safety "${pass_args[@]}" $SAFETY_SAMPLING "$@" || status=$?
fi

if [[ "$RUN_TRANSLATION" == "true" ]]; then
  echo
  echo "#########################  translation  #########################"
  rebuild_args "$TRANSLATION_BENCHMARKS"
  pass_args+=(
    --xcomet_model      "$XCOMET_MODEL"
    --xcomet_batch_size "$XCOMET_BATCH_SIZE"
    --xcomet_gpus       "$XCOMET_GPUS"
    --xcomet_cache_dir  "$XCOMET_CACHE_DIR"
    --xcomet            "$XCOMET"
  )
  # shellcheck disable=SC2086
  python -m translation "${pass_args[@]}" $TRANSLATION_SAMPLING "$@" || status=$?
fi

exit "$status"
