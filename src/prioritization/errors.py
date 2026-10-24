"""Exceptions raised by the bias-aware prioritization feature."""

from __future__ import annotations


class PrioritizationError(Exception):
    """Base class for all prioritization errors."""


class ConfigurationError(PrioritizationError):
    """Raised when artifact configuration is invalid."""


class ArtifactError(PrioritizationError):
    """Base class for model artifact loading and validation errors."""


class ArtifactNotFoundError(ArtifactError):
    """Raised when a required model artifact cannot be found."""


class ArtifactLoadError(ArtifactError):
    """Raised when an artifact exists but cannot be deserialized."""


class ArtifactValidationError(ArtifactError):
    """Raised when an artifact has an unexpected type or feature shape."""


class InputValidationError(ValueError, PrioritizationError):
    """Base class for invalid claims, ratings, or objectives."""


class InvalidObjectiveError(InputValidationError):
    """Raised when an objective other than O1, O2, or O3 is requested."""


class InvalidClaimError(InputValidationError):
    """Raised when a claim record is malformed."""


class DuplicateClaimError(InvalidClaimError):
    """Raised when more than one claim uses the same claim ID."""


class InvalidRatingError(InputValidationError):
    """Raised when a user rating is malformed or uses an unknown label."""


class UnknownClaimError(InvalidRatingError):
    """Raised when a rating references a claim ID that was not supplied."""


class MissingPerceptionError(InputValidationError):
    """Raised when one or more claims have no perception ratings."""


class InferenceError(RuntimeError, PrioritizationError):
    """Raised when artifact inference returns unusable output."""
