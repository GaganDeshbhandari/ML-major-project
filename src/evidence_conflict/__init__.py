"""Typed contracts, configuration, and pure ECS aggregation.

The package provides the public schemas, configuration, and errors used by the
evidence-conflict modules. NLI model loading and stance inference live in
``nli_classifier``; pure ECS aggregation lives in ``scoring``. The pipeline
integrates both behind an opt-in configuration. Importing this package does
not import PyTorch or Transformers, download models, or load weights.
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
