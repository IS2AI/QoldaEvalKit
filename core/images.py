"""Turning dataset images into data: URIs for the chat completions API."""

from __future__ import annotations

import base64
import io
from typing import Any, Optional

from .config import ImageConfig

_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


def encode_image(image: Any, config: ImageConfig) -> Optional[str]:
    """Encode a PIL image / raw bytes / path into a base64 data URI payload.

    Returns ``None`` when the image cannot be read, so the caller can drop the
    item rather than sending a malformed request.
    """
    if image is None:
        return None

    # Already-encoded bytes are passed through untouched unless a resize is
    # requested, since re-encoding only loses quality.
    if isinstance(image, (bytes, bytearray)):
        if not config.max_size:
            return base64.b64encode(bytes(image)).decode("utf-8")
        image = _open(io.BytesIO(bytes(image)))
    elif isinstance(image, dict) and "bytes" in image:
        # HF sometimes hands back {"bytes": ..., "path": ...}.
        return encode_image(image["bytes"], config)
    elif isinstance(image, str):
        try:
            image = _open(image)
        except Exception:  # noqa: BLE001
            return None

    if image is None or not hasattr(image, "save"):
        return None

    if config.max_size and max(image.size) > config.max_size:
        from PIL import Image as _Image

        scale = config.max_size / max(image.size)
        new_size = (max(1, int(image.width * scale)),
                    max(1, int(image.height * scale)))
        # LANCZOS: downscaling by a large ratio with the default filter loses
        # detail that matters for chart and document questions.
        image = image.resize(new_size, _Image.LANCZOS)

    fmt = config.format.upper()
    if fmt in ("JPEG", "WEBP") and image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    buffer = io.BytesIO()
    save_kwargs = {"quality": config.quality} if fmt in ("JPEG", "WEBP") else {}
    image.save(buffer, format=fmt, **save_kwargs)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _open(source: Any):
    from PIL import Image

    return Image.open(source)


def data_uri(encoded: str, config: ImageConfig) -> str:
    mime = _MIME.get(config.format.upper(), "image/jpeg")
    return f"data:{mime};base64,{encoded}"
