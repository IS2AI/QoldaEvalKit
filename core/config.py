"""Configuration objects shared by every modality.

Everything the CLI can tune lives here as a dataclass so that a run is fully
described by ``RunConfig`` and can be serialised into ``run_config.json`` next
to the results.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Languages the toolkit knows about. Kazakh is always the primary one.
LANGUAGES = ("kk", "ru", "en")

LANGUAGE_NAMES = {
    "kk": "Kazakh",
    "ru": "Russian",
    "en": "English",
}


@dataclass
class SamplingConfig:
    """Decoding parameters forwarded to the vLLM OpenAI-compatible server."""

    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = 16384
    seed: Optional[int] = None

    def split_kwargs(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Split params into OpenAI-standard kwargs and vLLM ``extra_body``.

        ``top_k``, ``min_p`` and ``repetition_penalty`` are vLLM extensions and
        are rejected by the stock OpenAI schema, so they travel in extra_body.
        """
        standard: Dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            standard["seed"] = self.seed

        extra: Dict[str, Any] = {}
        # Only send sampler knobs that actually deviate from the neutral value,
        # so servers that do not implement them are not needlessly upset.
        if self.top_k and self.top_k > 0:
            extra["top_k"] = self.top_k
        if self.min_p and self.min_p > 0:
            extra["min_p"] = self.min_p
        if self.repetition_penalty and self.repetition_penalty != 1.0:
            extra["repetition_penalty"] = self.repetition_penalty
        return standard, extra


@dataclass
class ImageConfig:
    """How dataset images are turned into data: URIs for the request."""

    format: str = "JPEG"
    # Longest edge in pixels; 0 disables resizing. Capped by default: dynamic-
    # resolution VLMs turn pixels into vision tokens with no upper bound.
    max_size: int = 1024
    quality: int = 90
    detail: Optional[str] = None   # "low" | "high" | "auto"; None omits the field


@dataclass
class AudioConfig:
    """How dataset audio is turned into a data: URI for the request."""

    sampling_rate: int = 16000
    format: str = "WAV"
    content_type: str = "audio_url"      # or "input_audio" (OpenAI schema)
    placeholder: str = ""                # e.g. "<audio>", if the template needs one
    max_seconds: float = 0.0             # 0 keeps the full recording


@dataclass
class TranslationConfig:
    """Neural MT metric settings (XCOMET runs locally, on a GPU)."""

    model: str = "Unbabel/XCOMET-XXL"
    batch_size: int = 8
    gpus: int = 1
    cache_dir: str = ""                  # empty uses comet's default
    # False saves translations unscored, so generation and GPU scoring can be
    # run separately.
    enabled: bool = True


@dataclass
class ModelConfig:
    """How to reach the served model and how to treat its reasoning traces."""

    api_base: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model_name: str = "model"
    system_prompt: str = ""

    thinking: str = "auto"               # on | off | auto (leave template alone)
    think_start_token: str = "<think>"
    think_end_token: str = "</think>"

    concurrency: int = 16
    retries: int = 5
    request_timeout: float = 1800.0

    # Gemma-3 / InternVL / DeepSeek-VL reject a system turn beside an image.
    no_system_role: bool = False

    # Guided decoding. Off for reasoning models: it forces the first token to
    # be "{", so <think> can never open. IFBench and the safety probes opt out
    # regardless, since the shape of the reply is what they measure.
    structured_output: bool = False

    @property
    def enable_thinking(self) -> Optional[bool]:
        if self.thinking == "on":
            return True
        if self.thinking == "off":
            return False
        return None  # "auto" -> do not touch chat_template_kwargs


@dataclass
class JudgeConfig:
    """LLM-as-a-judge settings (OpenAI Batch API, strict 0/1 output)."""

    model: str = "gpt-5.6-luna"
    api_key: str = field(default_factory=lambda: os.getenv("JUDGE_API_KEY", ""))
    base_url: Optional[str] = None
    reasoning_effort: str = "none"
    # Covers reasoning tokens as well as the reply, so it moves with the effort
    # above: "medium" needs roughly 512 here or the reply comes back empty.
    max_completion_tokens: int = 32
    completion_window: str = "24h"
    poll_interval: float = 30.0

    direct_below: int = 500              # smaller sets skip the Batch API
    # Judge requests in flight across all direct jobs. 1 = strictly sequential.
    direct_concurrency: int = 1
    # Tries per direct request. An abandoned item leaves the denominator
    # entirely rather than scoring 0, so this is deliberately generous.
    direct_max_attempts: int = 2
    split_above: int = 5000              # at 5000, only RAGBench splits
    split_parts: int = 3
    max_parallel: int = 4                # batch jobs in flight at once
    structured: bool = True              # strict JSON schema for the verdict
    # Hard ceiling on how long we block waiting for a batch (seconds).
    timeout: float = 24 * 60 * 60
    enabled: bool = True


@dataclass
class RunConfig:
    """A complete evaluation run."""

    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)

    modality: str = "text"
    run_name: str = "run"
    output_root: str = "results"
    data_dir: str = "datasets"
    # Where a user-supplied ASR corpus lives (manifest + audio files). Empty
    # means "use the Hub copy of FLEURS instead".
    asr_data_dir: str = ""

    benchmarks: List[str] = field(default_factory=lambda: ["all"])
    # Ignore selected keys this modality does not own (used by MODALITY=all).
    skip_unknown_benchmarks: bool = False
    languages: List[str] = field(default_factory=lambda: ["kk"])

    data_portion: float = 1.0
    seed: int = 42

    # Persist the model's final answer (and its reasoning trace) per item.
    save_responses: bool = True
    # Resume: skip items already present in the records file.
    resume: bool = False
    # Discard existing judge verdicts and grade again. Needed after changing
    # the judge model, prompt or effort; generation is still reused.
    rejudge: bool = False

    # Extra passes over items no answer could be parsed from. Only empty
    # predictions are retried, never wrong ones, so it cannot inflate a score.
    retry_unparsed: int = 1

    # "en" keeps harness instructions English while the question stays in its
    # own language; "auto" matches the instructions to the benchmark.
    prompt_lang: str = "en"

    @property
    def data_dirs(self) -> Dict[str, str]:
        """Directory per Source.dir_key. An empty "asr" falls back to the Hub."""
        return {"data": self.data_dir, "asr": self.asr_data_dir}

    @property
    def model_dir(self) -> str:
        """Everything produced for one model, across every modality."""
        return os.path.join(self.output_root, self.run_name)

    @property
    def run_dir(self) -> str:
        return os.path.join(self.model_dir, self.modality)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Never write credentials into the results directory.
        d["model"]["api_key"] = "***" if self.model.api_key else ""
        d["judge"]["api_key"] = "***" if self.judge.api_key else ""
        return d
