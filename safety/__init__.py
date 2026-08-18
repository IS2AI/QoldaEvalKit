"""Qorgau safety benchmark for Kazakh and Russian."""

from .modality import SafetyModality
from .specs import REGISTRY

__all__ = ["REGISTRY", "SafetyModality"]
