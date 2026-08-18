"""Async generation against a vLLM OpenAI-compatible endpoint."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence

from .config import AudioConfig, ImageConfig, ModelConfig, SamplingConfig
from .extraction import Completion, reasoning_from_message, split_reasoning
from .audio import data_uri as audio_data_uri
from .images import data_uri

logger = logging.getLogger("qoldaevalkit")

# VLM families that reject a system turn alongside multimodal content.
NO_SYSTEM_ROLE_MODELS = (
    "gemma-3", "gemma3", "internvl", "intern-vl",
    "internlm-xcomposer", "deepseek-vl",
)


def needs_no_system_role(model_name: str) -> bool:
    lowered = model_name.lower()
    return any(marker in lowered for marker in NO_SYSTEM_ROLE_MODELS)


class ModelClient:
    """Thin wrapper that owns concurrency, retries and reasoning handling."""

    def __init__(self, model: ModelConfig, sampling: SamplingConfig,
                 image: Optional[ImageConfig] = None,
                 audio: Optional[AudioConfig] = None):
        from openai import AsyncOpenAI

        self.model = model
        self.sampling = sampling
        self.image = image or ImageConfig()
        self.audio = audio or AudioConfig()
        client_kwargs: Dict[str, Any] = {
            "api_key": model.api_key or "EMPTY",
            "base_url": model.api_base,
            "timeout": model.request_timeout,
            "max_retries": 0,  # retries are handled here so they can be logged
        }
        # httpx keeps only 20 idle connections by default, so a run at
        # --concurrency 64 would spend its time reopening sockets. Size the
        # pool to the concurrency instead.
        try:
            import httpx

            client_kwargs["http_client"] = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=max(64, model.concurrency * 2),
                    max_keepalive_connections=max(32, model.concurrency),
                ),
                timeout=model.request_timeout,
            )
        except Exception:  # noqa: BLE001 - fall back to the library default
            pass
        self._client = AsyncOpenAI(**client_kwargs)

        # Secondary gate for callers using the client directly; the runner
        # gates the whole item pipeline, which is the limit that matters.
        self._semaphore = asyncio.Semaphore(model.concurrency)

    def _audio_part(self, encoded: str) -> Dict[str, Any]:
        if self.audio.content_type == "input_audio":
            return {"type": "input_audio",
                    "input_audio": {"data": encoded,
                                    "format": self.audio.format.lower()}}
        return {"type": "audio_url",
                "audio_url": {"url": audio_data_uri(encoded, self.audio)}}

    def _build_messages(self, prompt: str, images: Optional[Sequence[str]],
                        audios: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
        prompt = prompt.strip()
        system = self.model.system_prompt

        if not images and not audios:
            messages: List[Dict[str, Any]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            return messages

        fold_system = system and self.model.no_system_role
        user_text = f"{system}\n\n{prompt}" if fold_system else prompt
        if audios and self.audio.placeholder:
            # Only for chat templates that do not insert their own marker.
            user_text = f"{self.audio.placeholder}\n{user_text}"

        content: List[Dict[str, Any]] = []
        for encoded in images or ():
            image_url: Dict[str, Any] = {"url": data_uri(encoded, self.image)}
            if self.image.detail:
                image_url["detail"] = self.image.detail
            content.append({"type": "image_url", "image_url": image_url})
        for encoded in audios or ():
            content.append(self._audio_part(encoded))
        content.append({"type": "text", "text": user_text})

        messages = []
        if system and not fold_system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        return messages

    def _request_kwargs(self, prompt: str, images: Optional[Sequence[str]],
                        audios: Optional[Sequence[str]],
                        schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        standard, extra_body = self.sampling.split_kwargs()

        enable_thinking = self.model.enable_thinking
        if enable_thinking is not None:
            # Reasoning-native models read this from their chat template.
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": enable_thinking
            }

        kwargs: Dict[str, Any] = {
            "model": self.model.model_name,
            "messages": self._build_messages(prompt, images, audios),
            **standard,
        }
        # vLLM accepts the OpenAI response_format shape and enforces it with
        # guided decoding. Only sent when the run asked for it AND the task
        # supplied a schema.
        if schema is not None and self.model.structured_output:
            kwargs["response_format"] = schema

        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    async def generate(self, prompt: str,
                       images: Optional[Sequence[str]] = None,
                       audios: Optional[Sequence[str]] = None,
                       schema: Optional[Dict[str, Any]] = None) -> Completion:
        """One completion, with bounded concurrency and retries."""
        async with self._semaphore:
            delay = 1.0
            last_error: Optional[Exception] = None
            for attempt in range(1, self.model.retries + 1):
                try:
                    response = await self._client.chat.completions.create(
                        **self._request_kwargs(prompt, images, audios, schema)
                    )
                except Exception as exc:  # noqa: BLE001 - retry anything
                    last_error = exc
                    if attempt < self.model.retries:
                        logger.warning("Generation failed (attempt %d/%d): %s",
                                       attempt, self.model.retries, exc)
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 60.0)
                        continue
                    break

                choice = response.choices[0]
                message = choice.message
                completion = split_reasoning(
                    content=message.content or "",
                    reasoning_content=reasoning_from_message(message),
                    think_start=self.model.think_start_token,
                    think_end=self.model.think_end_token,
                )
                completion.finish_reason = choice.finish_reason or ""
                if completion.finish_reason == "length" and not completion.answer:
                    completion.truncated_thinking = True
                return completion

            logger.error("Generation failed after %d attempts: %s",
                         self.model.retries, last_error)
            return Completion(finish_reason="error")

    async def close(self) -> None:
        await self._client.close()
