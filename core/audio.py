"""Turning dataset audio into data: URIs for the chat completions API."""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Optional

# The loader takes clips undecoded (Audio(decode=False)) and they are decoded
# here with soundfile, so torchcodec is never required. This env var is a
# no-op on datasets>=5 and is kept only for older installs.
os.environ.setdefault("HF_AUDIO_DECODER", "soundfile")

from .config import AudioConfig  # noqa: E402

_MIME = {"WAV": "audio/wav", "FLAC": "audio/flac", "OGG": "audio/ogg",
         "MP3": "audio/mpeg"}


def encode_audio(audio: Any, config: AudioConfig) -> Optional[str]:
    """Encode a decoded HF audio value into base64 mono audio at one rate.

    Accepts what the ``datasets`` Audio feature yields — ``{"array", "sampling_rate"}``
    or ``{"bytes"|"path"}`` — plus raw bytes and file paths.  Returns ``None``
    when the clip cannot be read, so the caller can drop the item rather than
    send a malformed request.
    """
    if audio is None:
        return None

    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The audio modality needs soundfile and numpy: pip install -r requirements.txt"
        ) from exc

    array = None
    rate = config.sampling_rate

    # datasets>=5 yields a torchcodec AudioDecoder when a column is left
    # decoded; unwrap it rather than failing.
    if hasattr(audio, "get_all_samples"):
        try:
            samples = audio.get_all_samples()
            audio = {"array": samples.data.squeeze().cpu().numpy(),
                     "sampling_rate": int(samples.sample_rate)}
        except Exception:  # noqa: BLE001
            return None

    try:
        if isinstance(audio, dict):
            if audio.get("array") is not None:
                array = np.asarray(audio["array"], dtype=np.float32)
                rate = int(audio.get("sampling_rate") or config.sampling_rate)
            elif audio.get("bytes"):
                array, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            elif audio.get("path"):
                array, rate = sf.read(str(audio["path"]), dtype="float32")
        elif isinstance(audio, (bytes, bytearray)):
            array, rate = sf.read(io.BytesIO(bytes(audio)), dtype="float32")
        elif isinstance(audio, (str, os.PathLike)):
            array, rate = sf.read(str(audio), dtype="float32")
        else:
            array = np.asarray(audio, dtype=np.float32)
    except Exception:  # noqa: BLE001
        return None

    if array is None or getattr(array, "size", 0) == 0:
        return None

    # Downmix to mono.
    if array.ndim > 1:
        array = array.mean(axis=1 if array.shape[0] > array.shape[1] else 0)
    array = np.asarray(array, dtype=np.float32).reshape(-1)

    if config.max_seconds:
        limit = int(config.max_seconds * rate)
        if limit and array.shape[0] > limit:
            array = array[:limit]

    if rate != config.sampling_rate:
        array = _resample(array, rate, config.sampling_rate)
        rate = config.sampling_rate

    buffer = io.BytesIO()
    subtype = "PCM_16" if config.format.upper() == "WAV" else None
    sf.write(buffer, array, rate, format=config.format.upper(), subtype=subtype)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _resample(array, source_rate: int, target_rate: int):
    """Resample to the model's rate, best available quality first.

    Benchmarks ship at 44.1/48 kHz and speech models want 16 kHz, so this runs
    on nearly every clip. Naive interpolation aliases badly enough to move WER,
    hence the preference order: soxr, then scipy's polyphase filter, and only
    then linear interpolation.
    """
    import numpy as np

    if source_rate == target_rate or source_rate <= 0:
        return array

    try:
        import soxr

        return soxr.resample(array, source_rate, target_rate).astype("float32")
    except ImportError:
        pass

    try:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(int(source_rate), int(target_rate))
        return resample_poly(array, target_rate // divisor,
                             source_rate // divisor).astype("float32")
    except ImportError:
        logger_warned = True  # noqa: F841 - fall through to linear

    duration = array.shape[0] / source_rate
    target_length = max(1, int(round(duration * target_rate)))
    source_positions = np.linspace(0.0, array.shape[0] - 1, num=array.shape[0])
    target_positions = np.linspace(0.0, array.shape[0] - 1, num=target_length)
    return np.interp(target_positions, source_positions, array).astype("float32")


def data_uri(encoded: str, config: AudioConfig) -> str:
    mime = _MIME.get(config.format.upper(), "audio/wav")
    return f"data:{mime};base64,{encoded}"
