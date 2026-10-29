"""Exceptions raised by the evidence conflict feature."""

from __future__ import annotations


class EvidenceConflictError(Exception):
    """Base class for all evidence conflict errors."""


class ECSConfigurationError(ValueError, EvidenceConflictError):
    """Raised when evidence conflict configuration is invalid."""


class StanceInputError(ValueError, EvidenceConflictError):
    """Raised when claim, evidence, or stance data is invalid."""


class StanceModelError(RuntimeError, EvidenceConflictError):
    """Raised when the stance model backend or weights cannot be loaded."""


class UnsupportedLabelMappingError(StanceModelError):
    """Raised when model labels cannot be mapped to stance labels."""


class StanceInferenceError(RuntimeError, EvidenceConflictError):
    """Raised when stance inference fails or returns invalid output."""


class ECSError(RuntimeError, EvidenceConflictError):
    """Raised when evidence conflict scoring fails."""
