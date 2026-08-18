"""FLORES translation benchmarks, scored with XCOMET."""

from .modality import TranslationModality
from .specs import REGISTRY

__all__ = ["REGISTRY", "TranslationModality"]
