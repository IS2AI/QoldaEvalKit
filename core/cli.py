"""Shared command-line plumbing for every modality."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import List, Optional

from .config import (
    AudioConfig,
    ImageConfig,
    JudgeConfig,
    ModelConfig,
    RunConfig,
    SamplingConfig,
    TranslationConfig,
)
from .generation import needs_no_system_role
from .registry import Registry
from .reporting import setup_logging
from .runner import Modality, run_evaluation


def boolean(value: str) -> bool:
    """Accept the shapes a bash script is likely to pass."""
    lowered = str(value).strip().lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def csv_list(value: str) -> List[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def build_parser(prog: str, description: str, with_images: bool,
                 with_audio: bool = False,
                 with_translation: bool = False,
                 presence_penalty: float = 1.5) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog, description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    served = parser.add_argument_group("served model")
    served.add_argument("--api_base", default="http://localhost:8000/v1",
                        help="Base URL of the vLLM OpenAI-compatible server.")
    served.add_argument("--api_key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    served.add_argument("--model_name", default=None,
                        help="Must match --served-model-name on the server.")
    served.add_argument("--run_name", default=None,
                        help="Results subdirectory. Defaults to --model_name.")
    served.add_argument("--system_prompt", default="")
    served.add_argument("--concurrency", type=int, default=16)
    served.add_argument("--retries", type=int, default=5)
    served.add_argument("--request_timeout", type=float, default=1800.0)

    thinking = parser.add_argument_group("reasoning / thinking")
    thinking.add_argument("--thinking", choices=["auto", "on", "off"],
                          default="auto",
                          help="'auto' leaves the chat template untouched "
                               "(correct for reasoning-native models); "
                               "'on'/'off' sets enable_thinking.")
    thinking.add_argument("--think_start_token", default="<think>")
    thinking.add_argument("--think_end_token", default="</think>")
    thinking.add_argument("--structured_output", type=boolean, default=False,
                          help="Constrain the evaluated model's reply with a "
                               "JSON schema. OFF by default: guided decoding "
                               "forces the first token to be '{', so a "
                               "reasoning model can never emit <think>.")
    thinking.add_argument("--save_responses", type=boolean, default=True,
                          help="Save the thinking trace and the final response "
                               "for every item.")

    sampling = parser.add_argument_group("sampling")
    sampling.add_argument("--temperature", type=float, default=1.0)
    sampling.add_argument("--top_p", type=float, default=0.95)
    sampling.add_argument("--top_k", type=int, default=20)
    sampling.add_argument("--min_p", type=float, default=0.0)
    # Vision defaults to 0.0: penalising repetition hurts descriptions and OCR.
    sampling.add_argument("--presence_penalty", type=float,
                          default=presence_penalty)
    sampling.add_argument("--repetition_penalty", type=float, default=1.0)
    sampling.add_argument("--max_tokens", type=int, default=16384)
    sampling.add_argument("--sampling_seed", type=int, default=None,
                          help="Server-side sampling seed (reproducible decoding).")

    if with_images:
        images = parser.add_argument_group("images")
        images.add_argument("--image_format", default="JPEG",
                            choices=["JPEG", "PNG", "WEBP"])
        images.add_argument("--image_max_size", type=int, default=1024,
                            help="Longest edge in pixels; 0 keeps the original. "
                                 "Capped by default: an unbounded image can "
                                 "produce tens of thousands of vision tokens on "
                                 "dynamic-resolution VLMs and stall the server.")
        images.add_argument("--image_quality", type=int, default=90)
        images.add_argument("--image_detail", default=None,
                            choices=["low", "high", "auto"],
                            help="OpenAI image detail hint; omitted by default.")
        images.add_argument("--no_system_role", type=boolean, default=None,
                            help="Fold the system prompt into the user turn. "
                                 "Auto-detected for Gemma-3 / InternVL / "
                                 "DeepSeek-VL when left unset.")

    if with_audio:
        audio = parser.add_argument_group("audio")
        audio.add_argument("--audio_sampling_rate", type=int, default=16000,
                           help="Every clip is resampled to this rate on decode.")
        audio.add_argument("--audio_format", default="WAV",
                           choices=["WAV", "FLAC", "OGG"])
        audio.add_argument("--audio_content_type", default="audio_url",
                           choices=["audio_url", "input_audio"],
                           help="Request shape: audio_url is what vLLM accepts; "
                                "input_audio is the OpenAI schema.")
        audio.add_argument("--audio_placeholder", default="",
                           help="Literal marker prefixed to the text turn, e.g. "
                                "'<audio>'. Most chat templates insert their "
                                "own, so this is empty by default.")
        audio.add_argument("--audio_max_seconds", type=float, default=0.0,
                           help="Truncate clips longer than this; 0 keeps them whole.")
        audio.add_argument("--asr_data_dir", default="",
                           help="Directory holding your ASR corpus (a TSV/CSV/"
                                "JSONL manifest plus its audio files, one "
                                "subdirectory per language). Left empty, ASR "
                                "reads FLEURS from the Hub instead.")

    if with_translation:
        mt = parser.add_argument_group("translation metric (XCOMET)")
        mt.add_argument("--xcomet_model", default="Unbabel/XCOMET-XXL",
                        help="Neural MT metric checkpoint. XCOMET-XXL is 10.7B "
                             "and gated on the Hub; Unbabel/XCOMET-XL or "
                             "Unbabel/wmt22-comet-da are lighter.")
        mt.add_argument("--xcomet_batch_size", type=int, default=8)
        mt.add_argument("--xcomet_gpus", type=int, default=1)
        mt.add_argument("--xcomet_cache_dir", default="",
                        help="Where to cache the checkpoint; comet's default "
                             "when empty.")
        mt.add_argument("--xcomet", type=boolean, default=True,
                        help="False saves the translations without scoring "
                             "them, so scoring can be done later on a GPU.")

    data = parser.add_argument_group("benchmarks and data")
    data.add_argument("--benchmarks", type=csv_list, default=["all"],
                      help="Comma-separated keys, or 'all'.")
    data.add_argument("--languages", type=csv_list, default=["kk"],
                      help="Comma-separated subset of kk,ru,en.")
    data.add_argument("--skip_unknown_benchmarks", type=boolean, default=False,
                      help="Ignore selected benchmark keys that belong to a "
                           "different modality instead of failing. Set "
                           "automatically when MODALITY=all.")
    data.add_argument("--data_portion", type=float, default=1.0)
    data.add_argument("--seed", type=int, default=42,
                      help="Seed for subsampling (not for decoding).")
    data.add_argument("--data_dir", default="datasets",
                      help="Directory holding the local JSONL benchmarks.")
    data.add_argument("--output_root", default="results")
    data.add_argument("--resume", type=boolean, default=False,
                      help="Skip items already present in the records files.")
    data.add_argument("--rejudge", type=boolean, default=False,
                      help="Discard verdicts a previous judge run produced and "
                           "grade them again. Use after changing the judge "
                           "model, prompt or reasoning effort; generation is "
                           "still reused.")
    data.add_argument("--retry_unparsed", type=int, default=1,
                      help="Extra generation passes over the items no answer "
                           "could be parsed from. Wrong answers are never "
                           "regenerated, only empty ones. 0 disables it.")
    data.add_argument("--prompt_lang", choices=["auto", "kk", "ru", "en"],
                      default="en",
                      help="Language of the instruction scaffolding wrapped "
                           "around each item. Default 'en': instructions are "
                           "English while the benchmark's own question stays "
                           "in its language. 'auto' matches the benchmark.")

    judge = parser.add_argument_group("LLM judge (OpenAI Batch API)")
    judge.add_argument("--judge", type=boolean, default=True,
                       help="Run the 0/1 batch judge where a benchmark needs it.")
    judge.add_argument("--judge_model", default="gpt-5.6-luna")
    judge.add_argument("--judge_api_key",
                       default=os.getenv("JUDGE_API_KEY")
                       or os.getenv("OPENAI_API_KEY", ""))
    judge.add_argument("--judge_base_url", default=os.getenv("JUDGE_BASE_URL"))
    judge.add_argument("--judge_reasoning_effort", default="none")
    judge.add_argument("--judge_max_completion_tokens", type=int, default=32,
                       help="Budget per judge reply. Covers reasoning tokens "
                            "too, so raise it alongside --judge_reasoning_"
                            "effort or the reply comes back empty.")
    judge.add_argument("--judge_completion_window", default="24h")
    judge.add_argument("--judge_poll_interval", type=float, default=30.0)
    judge.add_argument("--judge_timeout", type=float, default=24 * 60 * 60,
                       help="Seconds to wait for a batch before giving up.")
    judge.add_argument("--judge_structured", type=boolean, default=True,
                       help="Constrain the judge with a strict JSON schema so a "
                            "verdict can never come back malformed.")
    judge.add_argument("--judge_direct_below", type=int, default=500,
                       help="Sets smaller than this skip the Batch API.")
    judge.add_argument("--judge_direct_concurrency", type=int, default=1,
                       help="Requests in flight per direct (non-batch) job. "
                            "1 grades strictly one at a time. Note that "
                            "--judge_max_parallel direct jobs can overlap, so "
                            "the peak is the product of the two.")
    judge.add_argument("--judge_direct_max_attempts", type=int, default=2,
                       help="Tries per direct request before an item is left "
                            "ungraded. Backs off exponentially and honours the "
                            "server's Retry-After.")
    judge.add_argument("--judge_split_above", type=int, default=5000,
                       help="Sets larger than this are split into several jobs.")
    judge.add_argument("--judge_split_parts", type=int, default=3)
    judge.add_argument("--judge_max_parallel", type=int, default=4,
                       help="Batch jobs in flight at once, across all benchmarks.")

    parser.add_argument("--list_benchmarks", action="store_true",
                        help="Print the registry with language coverage and exit.")
    parser.add_argument("--verbose", action="store_true",
                        help="Mirror INFO logs to the console.")
    return parser


def print_registry(registry: Registry) -> None:
    width = max(len(key) for key in registry)
    print(f"\n{'benchmark'.ljust(width)}  {'langs':<10} {'task':<14} description")
    print("-" * 100)
    for key, spec in registry.items():
        langs = ",".join(spec.languages)
        tasks = sorted({spec.task_for(l).value for l in spec.languages})
        print(f"{key.ljust(width)}  {langs:<10} {'/'.join(tasks):<14} "
              f"{spec.description}")
    print()


def build_config(args, modality_name: str, with_images: bool,
                 with_audio: bool = False,
                 with_translation: bool = False) -> RunConfig:
    image = ImageConfig()
    audio = AudioConfig()
    if with_audio:
        audio = AudioConfig(
            sampling_rate=args.audio_sampling_rate,
            format=args.audio_format,
            content_type=args.audio_content_type,
            placeholder=args.audio_placeholder,
            max_seconds=args.audio_max_seconds,
        )
    no_system_role = needs_no_system_role(args.model_name)
    if with_images:
        image = ImageConfig(
            format=args.image_format,
            max_size=args.image_max_size,
            quality=args.image_quality,
            detail=args.image_detail,
        )
        if args.no_system_role is not None:
            no_system_role = args.no_system_role

    translation = TranslationConfig()
    if with_translation:
        translation = TranslationConfig(
            model=args.xcomet_model,
            batch_size=args.xcomet_batch_size,
            gpus=args.xcomet_gpus,
            cache_dir=args.xcomet_cache_dir,
            enabled=args.xcomet,
        )

    return RunConfig(
        model=ModelConfig(
            api_base=args.api_base,
            api_key=args.api_key,
            model_name=args.model_name,
            system_prompt=args.system_prompt,
            thinking=args.thinking,
            think_start_token=args.think_start_token,
            think_end_token=args.think_end_token,
            concurrency=args.concurrency,
            retries=args.retries,
            request_timeout=args.request_timeout,
            no_system_role=no_system_role,
            structured_output=args.structured_output,
        ),
        sampling=SamplingConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_tokens,
            seed=args.sampling_seed,
        ),
        judge=JudgeConfig(
            model=args.judge_model,
            api_key=args.judge_api_key or "",
            base_url=args.judge_base_url,
            reasoning_effort=args.judge_reasoning_effort,
            max_completion_tokens=args.judge_max_completion_tokens,
            completion_window=args.judge_completion_window,
            poll_interval=args.judge_poll_interval,
            timeout=args.judge_timeout,
            enabled=args.judge,
            structured=args.judge_structured,
            direct_below=args.judge_direct_below,
            direct_concurrency=max(1, args.judge_direct_concurrency),
            direct_max_attempts=max(1, args.judge_direct_max_attempts),
            split_above=args.judge_split_above,
            split_parts=args.judge_split_parts,
            max_parallel=args.judge_max_parallel,
        ),
        image=image,
        audio=audio,
        translation=translation,
        modality=modality_name,
        run_name=args.run_name or args.model_name,
        output_root=args.output_root,
        data_dir=args.data_dir,
        asr_data_dir=getattr(args, "asr_data_dir", "") or "",
        benchmarks=args.benchmarks,
        skip_unknown_benchmarks=args.skip_unknown_benchmarks,
        languages=args.languages,
        data_portion=args.data_portion,
        seed=args.seed,
        save_responses=args.save_responses,
        resume=args.resume,
        rejudge=args.rejudge,
        retry_unparsed=max(0, args.retry_unparsed),
        prompt_lang=args.prompt_lang,
    )


def print_banner(config: RunConfig, with_images: bool,
                 with_audio: bool = False,
                 with_translation: bool = False) -> None:
    print("=" * 72)
    print(f"modality     {config.modality}")
    print(f"model        {config.model.model_name}  @ {config.model.api_base}")
    print(f"languages    {', '.join(config.languages)}")
    print(f"benchmarks   {', '.join(config.benchmarks)}")
    print(f"sampling     temp={config.sampling.temperature} "
          f"top_p={config.sampling.top_p} top_k={config.sampling.top_k} "
          f"min_p={config.sampling.min_p} "
          f"pp={config.sampling.presence_penalty} "
          f"rp={config.sampling.repetition_penalty} "
          f"max_tokens={config.sampling.max_tokens}")
    print(f"thinking     {config.model.thinking}  "
          f"(save_responses={config.save_responses})")
    if with_images:
        print(f"images       {config.image.format} "
              f"max_size={config.image.max_size or 'native'} "
              f"quality={config.image.quality} "
              f"no_system_role={config.model.no_system_role}")
    if with_audio:
        print(f"asr corpus   {config.asr_data_dir or '<FLEURS from the Hub>'}")
        print(f"audio        {config.audio.format} @ {config.audio.sampling_rate}Hz "
              f"as={config.audio.content_type} "
              f"placeholder={config.audio.placeholder or '<none>'}")
    if with_translation:
        print(f"xcomet       {config.translation.model} "
              f"(gpus={config.translation.gpus}, "
              f"batch={config.translation.batch_size}, "
              f"enabled={config.translation.enabled})")
    print(f"judge        {config.judge.model} "
          f"(effort={config.judge.reasoning_effort}, "
          f"enabled={config.judge.enabled}, "
          f"schema={config.judge.structured})")
    print(f"             direct<{config.judge.direct_below} "
          f"split>{config.judge.split_above}/{config.judge.split_parts} "
          f"parallel={config.judge.max_parallel}")
    if config.model.structured_output:
        print("structured   ON — the model's reply is schema-constrained; "
              "reasoning traces will be empty")
    print(f"results      {config.run_dir}")
    print("=" * 72)


def _load_env() -> None:
    """Load .env from the toolkit directory, whatever the working directory.

    python-dotenv's search starts at the caller's cwd, which is not reliably
    the project root; the file sits next to run_eval.sh, so look there too.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()                                   # cwd and its parents
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(root, ".env"))         # next to run_eval.sh


def run_cli(modality: Modality, description: str, with_images: bool = False,
            with_audio: bool = False, with_translation: bool = False,
            argv: Optional[List[str]] = None) -> int:
    # .env must be loaded BEFORE the parser is built. Several arguments take
    # their default from os.getenv (--judge_api_key falls back to
    # OPENAI_API_KEY), and argparse evaluates defaults at construction time —
    # loading .env afterwards left the key empty and silently skipped every
    # grading job with "no judge API key".
    _load_env()

    parser = build_parser(f"{modality.name}_modality", description, with_images,
                          with_audio, with_translation,
                          presence_penalty=0.0 if modality.name == "vision" else 1.5)
    args = parser.parse_args(argv)

    if args.list_benchmarks:
        print_registry(modality.registry)
        return 0

    if not args.model_name:
        parser.error("--model_name is required (use --list_benchmarks to browse)")

    unknown = [l for l in args.languages if l not in ("kk", "ru", "en")]
    if unknown:
        parser.error(f"unknown language(s): {', '.join(unknown)}")

    config = build_config(args, modality.name, with_images, with_audio,
                          with_translation)

    # Fail loudly rather than discovering after hours of generation that every
    # judged benchmark was skipped for want of a key.
    if config.judge.enabled and not config.judge.api_key:
        judged = [s.key for s in modality.registry.values()
                  if any(s.task_for(l).is_judged for l in s.sources)]
        if judged:
            parser.error(
                "--judge is on but no judge API key was found. Set "
                "JUDGE_API_KEY or OPENAI_API_KEY in .env (next to run_eval.sh) "
                "or pass --judge_api_key. Benchmarks that need it: "
                f"{', '.join(judged)}. Use --judge false to run without them."
            )
    setup_logging(config.run_dir, verbose=args.verbose)
    print_banner(config, with_images, with_audio, with_translation)

    asyncio.run(run_evaluation(config, modality))
    return 0
