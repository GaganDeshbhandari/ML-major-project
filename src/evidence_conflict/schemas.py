"""Typed data structures for evidence conflict scoring.

This module defines contracts only. It does not load models, run inference, or
calculate an Evidence Conflict Score.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from math import isclose, isfinite
from numbers import Real

from .errors import StanceInputError

#: Absolute tolerance used when checking that three-class probabilities sum to 1.
_PROBABILITY_SUM_TOLERANCE = 1e-6


def _as_finite_float(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Return ``value`` as a finite float, enforcing an optional range."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise StanceInputError(
            f"{field} must be a real number; got {type(value).__name__}."
        )
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StanceInputError(
            f"{field} must be a finite real number; got {value!r}."
        ) from exc
    if not isfinite(number):
        raise StanceInputError(f"{field} must be finite; got {number!r}.")
    if minimum is not None and number < minimum:
        raise StanceInputError(f"{field} must be >= {minimum}; got {number!r}.")
    if maximum is not None and number > maximum:
        raise StanceInputError(f"{field} must be <= {maximum}; got {number!r}.")
    return number


def _as_nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StanceInputError(
            f"{field} must be an integer; got {type(value).__name__}."
        )
    if value < 0:
        raise StanceInputError(f"{field} must be >= 0; got {value!r}.")
    return value


def _require_bool(value: object, field: str) -> None:
    if not isinstance(value, bool):
        raise StanceInputError(
            f"{field} must be a boolean; got {type(value).__name__}."
        )


class StanceLabel(str, Enum):
    """The three possible stance labels for one evidence passage."""

    SUPPORTS = "supports"
    REFUTES = "refutes"
    NOT_ENOUGH_INFO = "not_enough_info"


@dataclass(frozen=True, slots=True)
class StanceProbabilities:
    """Three-class probabilities for one evidence passage."""

    supports: float
    refutes: float
    not_enough_info: float

    def __post_init__(self) -> None:
        for field in ("supports", "refutes", "not_enough_info"):
            value = _as_finite_float(
                getattr(self, field),
                field,
                minimum=0.0,
                maximum=1.0,
            )
            object.__setattr__(self, field, value)

        total = self.supports + self.refutes + self.not_enough_info
        if not isclose(
            total,
            1.0,
            rel_tol=0.0,
            abs_tol=_PROBABILITY_SUM_TOLERANCE,
        ):
            raise StanceInputError(
                "Stance probabilities must sum to approximately 1.0; "
                f"got {total:.6f}."
            )


@dataclass(frozen=True, slots=True)
class StancePrediction:
    """The stance prediction for one evidence passage.

    ``truncated`` records whether the input text was cut before inference.
    ``abstained`` records threshold-based abstention; the label and
    probabilities still describe the model output.
    """

    evidence_index: int
    label: StanceLabel
    confidence: float
    probabilities: StanceProbabilities
    truncated: bool = False
    abstained: bool = False

    def __post_init__(self) -> None:
        evidence_index = _as_nonnegative_int(
            self.evidence_index,
            "evidence_index",
        )
        object.__setattr__(self, "evidence_index", evidence_index)

        if not isinstance(self.label, StanceLabel):
            raise StanceInputError("label must be a StanceLabel.")

        confidence = _as_finite_float(
            self.confidence,
            "confidence",
            minimum=0.0,
            maximum=1.0,
        )
        object.__setattr__(self, "confidence", confidence)

        if not isinstance(self.probabilities, StanceProbabilities):
            raise StanceInputError(
                "probabilities must be a StanceProbabilities instance."
            )

        _require_bool(self.truncated, "truncated")
        _require_bool(self.abstained, "abstained")


@dataclass(frozen=True, slots=True)
class StanceCounts:
    """Counts of supporting, refuting, and inconclusive evidence passages."""

    supports: int = 0
    refutes: int = 0
    not_enough_info: int = 0

    def __post_init__(self) -> None:
        for field in ("supports", "refutes", "not_enough_info"):
            value = _as_nonnegative_int(getattr(self, field), field)
            object.__setattr__(self, field, value)

    @property
    def total(self) -> int:
        """Return the total number of counted passages."""

        return self.supports + self.refutes + self.not_enough_info


class ECSStatus(str, Enum):
    """Lifecycle status for an evidence conflict result."""

    DISABLED = "disabled"
    NO_EVIDENCE = "no_evidence"
    RETRIEVAL_ERROR = "retrieval_error"
    NLI_ERROR = "nli_error"
    INSUFFICIENT_STANCE = "insufficient_stance"
    COMPUTED = "computed"


def _counts_for_predictions(
    predictions: Iterable[StancePrediction],
) -> StanceCounts:
    supports = 0
    refutes = 0
    not_enough_info = 0
    for prediction in predictions:
        if prediction.label is StanceLabel.SUPPORTS:
            supports += 1
        elif prediction.label is StanceLabel.REFUTES:
            refutes += 1
        else:
            not_enough_info += 1
    return StanceCounts(
        supports=supports,
        refutes=refutes,
        not_enough_info=not_enough_info,
    )


@dataclass(frozen=True, slots=True)
class EvidenceConflictResult:
    """Structured evidence conflict output for one claim.

    ``ecs`` is required when ``status`` is :attr:`ECSStatus.COMPUTED` and must
    be ``None`` for every other status. Its scale is intentionally unconstrained
    in this phase. The counts must match the supplied predictions exactly.
    """

    claim: str
    predictions: tuple[StancePrediction, ...]
    counts: StanceCounts
    status: ECSStatus
    ecs: float | None = None
    evidence_coverage: float | None = None
    refutation_share: float | None = None
    support_share: float | None = None
    mean_confidence: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.claim, str) or not self.claim.strip():
            raise StanceInputError("claim must be a non-empty string.")

        try:
            predictions = tuple(self.predictions)
        except TypeError as exc:
            raise StanceInputError(
                "predictions must be an iterable of StancePrediction records."
            ) from exc
        if not all(isinstance(item, StancePrediction) for item in predictions):
            raise StanceInputError(
                "predictions must contain only StancePrediction records."
            )
        object.__setattr__(self, "predictions", predictions)

        if not isinstance(self.counts, StanceCounts):
            raise StanceInputError("counts must be a StanceCounts instance.")
        if not isinstance(self.status, ECSStatus):
            raise StanceInputError("status must be an ECSStatus.")

        if self.status is ECSStatus.COMPUTED:
            if self.ecs is None:
                raise StanceInputError(
                    "ecs must be set when status is COMPUTED."
                )
        elif self.ecs is not None:
            raise StanceInputError(
                "ecs must be None when status is not COMPUTED."
            )

        if self.ecs is not None:
            ecs = _as_finite_float(self.ecs, "ecs")
            object.__setattr__(self, "ecs", ecs)

        for field in (
            "evidence_coverage",
            "refutation_share",
            "support_share",
            "mean_confidence",
        ):
            value = getattr(self, field)
            if value is not None:
                normalized = _as_finite_float(
                    value,
                    field,
                    minimum=0.0,
                    maximum=1.0,
                )
                object.__setattr__(self, field, normalized)

        expected = _counts_for_predictions(predictions)
        if self.counts != expected:
            raise StanceInputError(
                "counts do not match predictions: "
                f"expected supports={expected.supports}, "
                f"refutes={expected.refutes}, "
                f"not_enough_info={expected.not_enough_info}; got "
                f"supports={self.counts.supports}, "
                f"refutes={self.counts.refutes}, "
                f"not_enough_info={self.counts.not_enough_info}."
            )
