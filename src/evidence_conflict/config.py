"""Immutable configuration for evidence conflict scoring.

This module only parses and validates configuration. It does not import
PyTorch or Transformers, probe devices, download models, or load weights.
Device selection is resolved by the inference layer in a later phase.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real

from .errors import ECSConfigurationError

#: Provisional default stance model. Model quality is evaluated in a later phase.
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
DEFAULT_DEVICE = "auto"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_EVIDENCE_CHARS = 2000

#: Practical upper guard; model-specific limits are enforced by the model layer.
MAX_SEQUENCE_LENGTH = 8192

ENV_NLI_MODEL = "ECS_NLI_MODEL"
ENV_DEVICE = "ECS_DEVICE"
ENV_MAX_LENGTH = "ECS_MAX_LENGTH"
ENV_BATCH_SIZE = "ECS_BATCH_SIZE"
ENV_CONFIDENCE_THRESHOLD = "ECS_CONFIDENCE_THRESHOLD"
ENV_MAX_EVIDENCE_CHARS = "ECS_MAX_EVIDENCE_CHARS"
ENV_LOCAL_FILES_ONLY = "ECS_LOCAL_FILES_ONLY"
ENV_MODEL_REVISION = "ECS_MODEL_REVISION"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def _env_value(env: Mapping[str, str], key: str) -> str | None:
    """Return a non-empty environment value, treating whitespace as unset."""

    value = env.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ECSConfigurationError(f"{key} must be a string.")
    stripped = value.strip()
    return stripped or None


def _parse_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = _env_value(env, key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ECSConfigurationError(
            f"{key} must be an integer; got {raw!r}."
        ) from exc


def _parse_optional_float(
    env: Mapping[str, str],
    key: str,
) -> float | None:
    raw = _env_value(env, key)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ECSConfigurationError(
            f"{key} must be a number; got {raw!r}."
        ) from exc


def _parse_bool(
    env: Mapping[str, str],
    key: str,
    default: bool,
) -> bool:
    raw = _env_value(env, key)
    if raw is None:
        return default
    normalized = raw.lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ECSConfigurationError(
        f"{key} must be one of: true/false, yes/no, on/off, 1/0; "
        f"got {raw!r}."
    )


def _require_int(
    value: object,
    field: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ECSConfigurationError(
            f"{field} must be an integer; got {type(value).__name__}."
        )
    if value < minimum:
        raise ECSConfigurationError(f"{field} must be >= {minimum}; got {value}.")
    if maximum is not None and value > maximum:
        raise ECSConfigurationError(
            f"{field} must be <= {maximum}; got {value}."
        )
    return value


def _require_fraction(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ECSConfigurationError(
            f"{field} must be a real number; got {type(value).__name__}."
        )
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ECSConfigurationError(
            f"{field} must be a finite real number; got {value!r}."
        ) from exc
    if not isfinite(number):
        raise ECSConfigurationError(f"{field} must be finite; got {number!r}.")
    if not 0.0 <= number <= 1.0:
        raise ECSConfigurationError(
            f"{field} must be in [0, 1]; got {number!r}."
        )
    return number


def _normalize_device(value: object) -> str:
    if not isinstance(value, str):
        raise ECSConfigurationError(
            f"device must be a string; got {type(value).__name__}."
        )
    normalized = value.strip().lower()
    if normalized in {"auto", "cpu", "cuda"}:
        return normalized
    if normalized.startswith("cuda:"):
        index = normalized[5:]
        if index.isascii() and index.isdigit():
            return normalized
    raise ECSConfigurationError(
        "device must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'; "
        f"got {value!r}."
    )


@dataclass(frozen=True, slots=True)
class ECSConfig:
    """Immutable runtime configuration for the evidence conflict feature.

    ``device='auto'`` means the inference layer should prefer CUDA when
    available and fall back to CPU. No device probing happens here; device
    availability is checked by the inference layer.

    ``max_length`` is a token limit. ``max_evidence_chars`` is a character cap
    applied before tokenization. ``confidence_threshold=None`` preserves pure
    argmax behavior; a numeric threshold is applied during inference.
    """

    model_name: str = DEFAULT_NLI_MODEL
    device: str = DEFAULT_DEVICE
    max_length: int = DEFAULT_MAX_LENGTH
    batch_size: int = DEFAULT_BATCH_SIZE
    confidence_threshold: float | None = None
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS
    local_files_only: bool = False
    model_revision: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ECSConfigurationError(
                "model_name must be a non-empty string."
            )
        object.__setattr__(self, "model_name", self.model_name.strip())

        object.__setattr__(self, "device", _normalize_device(self.device))

        max_length = _require_int(
            self.max_length,
            "max_length",
            minimum=1,
            maximum=MAX_SEQUENCE_LENGTH,
        )
        object.__setattr__(self, "max_length", max_length)

        batch_size = _require_int(
            self.batch_size,
            "batch_size",
            minimum=1,
        )
        object.__setattr__(self, "batch_size", batch_size)

        if self.confidence_threshold is not None:
            threshold = _require_fraction(
                self.confidence_threshold,
                "confidence_threshold",
            )
            object.__setattr__(self, "confidence_threshold", threshold)

        max_evidence_chars = _require_int(
            self.max_evidence_chars,
            "max_evidence_chars",
            minimum=1,
        )
        object.__setattr__(self, "max_evidence_chars", max_evidence_chars)

        if not isinstance(self.local_files_only, bool):
            raise ECSConfigurationError(
                "local_files_only must be a boolean."
            )

        if self.model_revision is not None:
            if not isinstance(self.model_revision, str):
                raise ECSConfigurationError(
                    "model_revision must be a non-empty string or None."
                )
            revision = self.model_revision.strip()
            object.__setattr__(self, "model_revision", revision or None)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ECSConfig:
        """Build configuration from optional ``ECS_*`` environment values.

        Environment variables are read only when this method is called. Blank
        values are treated as unset. The supplied mapping is never mutated.
        """

        values = os.environ if env is None else env
        if not isinstance(values, Mapping):
            raise ECSConfigurationError(
                "env must be a mapping of strings to strings."
            )

        return cls(
            model_name=(
                _env_value(values, ENV_NLI_MODEL) or DEFAULT_NLI_MODEL
            ),
            device=_env_value(values, ENV_DEVICE) or DEFAULT_DEVICE,
            max_length=_parse_int(
                values,
                ENV_MAX_LENGTH,
                DEFAULT_MAX_LENGTH,
            ),
            batch_size=_parse_int(
                values,
                ENV_BATCH_SIZE,
                DEFAULT_BATCH_SIZE,
            ),
            confidence_threshold=_parse_optional_float(
                values,
                ENV_CONFIDENCE_THRESHOLD,
            ),
            max_evidence_chars=_parse_int(
                values,
                ENV_MAX_EVIDENCE_CHARS,
                DEFAULT_MAX_EVIDENCE_CHARS,
            ),
            local_files_only=_parse_bool(
                values,
                ENV_LOCAL_FILES_ONLY,
                False,
            ),
            model_revision=_env_value(values, ENV_MODEL_REVISION),
        )
