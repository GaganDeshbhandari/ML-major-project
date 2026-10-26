"""Typed contracts and configuration for evidence conflict scoring.

This package currently provides schemas, configuration, and errors only. NLI
model loading, stance inference, ECS calculation, and pipeline integration are
intentionally left to later phases. Importing this package does not import
PyTorch or Transformers, download models, or load weights.
"""

from .config import DEFAULT_NLI_MODEL, ECSConfig
from .errors import (
    ECSConfigurationError,
    ECSError,
    EvidenceConflictError,
    StanceInferenceError,
    StanceInputError,
    StanceModelError,
    UnsupportedLabelMappingError,
)
from .schemas import (
    ECSStatus,
    EvidenceConflictResult,
    StanceCounts,
    StanceLabel,
    StancePrediction,
    StanceProbabilities,
)

__all__ = [
    "DEFAULT_NLI_MODEL",
    "ECSConfig",
    "ECSConfigurationError",
    "ECSError",
    "ECSStatus",
    "EvidenceConflictError",
    "EvidenceConflictResult",
    "StanceCounts",
    "StanceInferenceError",
    "StanceInputError",
    "StanceLabel",
    "StanceModelError",
    "StancePrediction",
    "StanceProbabilities",
    "UnsupportedLabelMappingError",
]
