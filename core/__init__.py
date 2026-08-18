"""Modality-agnostic core of QoldaEvalKit.

Data model, dataset loading, generation, scoring backends and reporting are
shared; ``text_modality`` and ``vision_modality`` supply only the prompts,
the benchmark specs and the scoring rules that differ between them.
"""

from .config import (
    ImageConfig,
    JudgeConfig,
    ModelConfig,
    RunConfig,
    SamplingConfig,
)
from .registry import BenchmarkSpec, Registry, Sample, Source, TaskType

__all__ = [
    "BenchmarkSpec",
    "ImageConfig",
    "JudgeConfig",
    "ModelConfig",
    "Registry",
    "RunConfig",
    "Sample",
    "SamplingConfig",
    "Source",
    "TaskType",
]

__version__ = "0.2.0"
