"""Pure Evidence Conflict Score (ECS) aggregation for one claim.

ECS measures inter-evidence stance conflict. It is a proposed project
extension, not a score defined by the original FactLLaMA paper, and it is not a
truth probability or a probability that the claim is false.

For one claim with ``n_total`` stance predictions:

- ``coverage = n_stance / n_total``
- ``ecs = 2 * min(n_supports, n_refutes) / n_total`` when stance evidence exists
- ``support_share = n_supports / n_stance``
- ``refutation_share = n_refutes / n_stance``

where ``n_stance = n_supports + n_refutes``. ``coverage`` is the proportion of
predictions that carry a stance (SUPPORTS or REFUTES). ``refutation_share`` is
the proportion of stance-bearing predictions that refute the claim. Undefined
values are represented by ``None`` and are never converted to zero.

Empty predictions produce ``NO_EVIDENCE``. Predictions that are all
NOT_ENOUGH_INFO produce ``INSUFFICIENT_STANCE``. One-sided stance evidence is
``COMPUTED`` with an ECS of ``0.0``.

``mean_confidence`` is the mean of the model's per-prediction confidence values
over all predictions. It is mean model confidence, not a calibrated probability
that a prediction or the resulting ECS is correct.

This module is pure: it does not load a model, run inference, access evidence
dictionaries, or handle retrieval/NLI failures. The score does not yet account
for source credibility, source independence, or calibration. It is not
empirically validated by the individual NLI stance benchmark; that benchmark
evaluates stance predictions, not this aggregation formula.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import fsum, isclose
from typing import Final

from .errors import ECSError, StanceInputError
from .schemas import (
    ECSStatus,
    EvidenceConflictResult,
    StanceCounts,
    StanceLabel,
    StancePrediction,
)

__all__ = ["compute_evidence_conflict"]

_FLOAT_TOLERANCE: Final = 1e-12


def compute_evidence_conflict(
    predictions: Iterable[StancePrediction],
    claim_text: str,
) -> EvidenceConflictResult:
    """Aggregate one claim's stance predictions into an ECS result.

    Args:
        predictions: Stance predictions in evidence order. The iterable is
            consumed once and is never mutated.
        claim_text: Non-empty claim text. It is stored unchanged in the result.

    Returns:
        A validated :class:`EvidenceConflictResult` with stance counts, status,
        ECS, evidence coverage, support/refutation shares, and mean model
        confidence. Prediction order is preserved.

    Raises:
        StanceInputError: If ``claim_text`` or ``predictions`` are invalid.
        ECSError: If an internal aggregate invariant fails.

    The function does not run inference or inspect evidence dictionaries.
    ``mean_confidence`` is the mean of all prediction confidences and is not a
    calibrated correctness probability.
    """

    claim = _validate_claim_text(claim_text)
    prediction_tuple = _validate_predictions(predictions)
    counts = _count_predictions(prediction_tuple)

    n_total = len(prediction_tuple)
    n_stance = counts.supports + counts.refutes

    if n_total == 0:
        status = ECSStatus.NO_EVIDENCE
        ecs = None
        coverage = None
        support_share = None
        refutation_share = None
        mean_confidence = None
    elif n_stance == 0:
        status = ECSStatus.INSUFFICIENT_STANCE
        ecs = None
        coverage = 0.0
        support_share = None
        refutation_share = None
        mean_confidence = _mean_confidence(prediction_tuple)
    else:
        status = ECSStatus.COMPUTED
        ecs = 2.0 * min(counts.supports, counts.refutes) / n_total
        coverage = n_stance / n_total
        support_share = counts.supports / n_stance
        refutation_share = counts.refutes / n_stance
        mean_confidence = _mean_confidence(prediction_tuple)

    result = EvidenceConflictResult(
        claim=claim,
        predictions=prediction_tuple,
        counts=counts,
        status=status,
        ecs=ecs,
        evidence_coverage=coverage,
        refutation_share=refutation_share,
        support_share=support_share,
        mean_confidence=mean_confidence,
    )
    _validate_result_invariants(result, prediction_tuple)
    return result


def _validate_claim_text(claim_text: object) -> str:
    if not isinstance(claim_text, str) or not claim_text.strip():
        raise StanceInputError("claim_text must be a non-empty string.")
    return claim_text


def _validate_predictions(
    predictions: Iterable[StancePrediction],
) -> tuple[StancePrediction, ...]:
    if isinstance(predictions, (str, bytes, bytearray)):
        raise StanceInputError(
            "predictions must be an iterable of StancePrediction records, "
            "not a string or byte sequence."
        )
    if isinstance(predictions, Mapping):
        raise StanceInputError(
            "predictions must be an iterable of StancePrediction records, "
            "not a mapping."
        )
    try:
        records = tuple(predictions)
    except TypeError as exc:
        raise StanceInputError(
            "predictions must be an iterable of StancePrediction records."
        ) from exc

    for index, prediction in enumerate(records):
        if not isinstance(prediction, StancePrediction):
            raise StanceInputError(
                f"predictions[{index}] must be a StancePrediction; "
                f"got {type(prediction).__name__}."
            )
        if not isinstance(prediction.label, StanceLabel):
            raise StanceInputError(
                f"predictions[{index}].label must be a StanceLabel; "
                f"got {prediction.label!r}."
            )
    return records


def _count_predictions(
    predictions: tuple[StancePrediction, ...],
) -> StanceCounts:
    supports = 0
    refutes = 0
    not_enough_info = 0
    for index, prediction in enumerate(predictions):
        if prediction.label is StanceLabel.SUPPORTS:
            supports += 1
        elif prediction.label is StanceLabel.REFUTES:
            refutes += 1
        elif prediction.label is StanceLabel.NOT_ENOUGH_INFO:
            not_enough_info += 1
        else:  # pragma: no cover - guarded by _validate_predictions
            raise StanceInputError(
                f"predictions[{index}].label is not a supported StanceLabel: "
                f"{prediction.label!r}."
            )
    return StanceCounts(
        supports=supports,
        refutes=refutes,
        not_enough_info=not_enough_info,
    )


def _mean_confidence(
    predictions: tuple[StancePrediction, ...],
) -> float:
    return fsum(prediction.confidence for prediction in predictions) / len(
        predictions
    )


def _validate_result_invariants(
    result: EvidenceConflictResult,
    expected_predictions: tuple[StancePrediction, ...],
) -> None:
    """Check that counts, status, and aggregate values agree.

    The result schema already validates counts against predictions and the
    presence of ECS for ``COMPUTED``. These checks are an independent guard
    around the aggregation formula and its status rules.
    """

    if result.predictions != expected_predictions:
        raise ECSError("prediction order was not preserved.")

    counts = _count_predictions(result.predictions)
    if result.counts != counts:
        raise ECSError(
            "counts do not match predictions after aggregation: "
            f"expected {counts!r}, got {result.counts!r}."
        )

    n_total = len(result.predictions)
    n_stance = counts.supports + counts.refutes
    if counts.total != n_total:
        raise ECSError(
            "counts.total must equal the number of predictions: "
            f"got {counts.total} and {n_total}."
        )

    if n_total == 0:
        expected_status = ECSStatus.NO_EVIDENCE
    elif n_stance == 0:
        expected_status = ECSStatus.INSUFFICIENT_STANCE
    else:
        expected_status = ECSStatus.COMPUTED
    if result.status is not expected_status:
        raise ECSError(
            f"status {result.status.value!r} does not match counts; "
            f"expected {expected_status.value!r}."
        )

    expected_coverage = None if n_total == 0 else n_stance / n_total
    expected_ecs = (
        None
        if n_total == 0 or n_stance == 0
        else 2.0 * min(counts.supports, counts.refutes) / n_total
    )
    expected_support_share = (
        None if n_stance == 0 else counts.supports / n_stance
    )
    expected_refutation_share = (
        None if n_stance == 0 else counts.refutes / n_stance
    )
    expected_mean_confidence = (
        None if n_total == 0 else _mean_confidence(result.predictions)
    )

    _require_close_or_none(
        result.ecs,
        expected_ecs,
        "ecs",
    )
    _require_close_or_none(
        result.evidence_coverage,
        expected_coverage,
        "evidence_coverage",
    )
    _require_close_or_none(
        result.support_share,
        expected_support_share,
        "support_share",
    )
    _require_close_or_none(
        result.refutation_share,
        expected_refutation_share,
        "refutation_share",
    )
    _require_close_or_none(
        result.mean_confidence,
        expected_mean_confidence,
        "mean_confidence",
    )

    for field in (
        "ecs",
        "evidence_coverage",
        "support_share",
        "refutation_share",
        "mean_confidence",
    ):
        value = getattr(result, field)
        if value is not None and not 0.0 <= value <= 1.0:
            raise ECSError(f"{field} must be in [0, 1]; got {value!r}.")

    if (
        expected_support_share is not None
        and expected_refutation_share is not None
        and not isclose(
            expected_support_share + expected_refutation_share,
            1.0,
            rel_tol=0.0,
            abs_tol=_FLOAT_TOLERANCE,
        )
    ):
        raise ECSError(
            "support_share and refutation_share must sum to 1.0 when "
            "stance evidence exists."
        )


def _require_close_or_none(
    actual: float | None,
    expected: float | None,
    field: str,
) -> None:
    if expected is None:
        if actual is not None:
            raise ECSError(f"{field} must be None for this status.")
        return
    if actual is None:
        raise ECSError(f"{field} must be defined for this status.")
    if not isclose(
        float(actual),
        float(expected),
        rel_tol=0.0,
        abs_tol=_FLOAT_TOLERANCE,
    ):
        raise ECSError(
            f"{field} does not match the aggregation formula: "
            f"expected {expected!r}, got {actual!r}."
        )
