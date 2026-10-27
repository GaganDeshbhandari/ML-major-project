"""Tests for evidence conflict schemas and validation."""

from __future__ import annotations

import math

import pytest

from src.evidence_conflict import (
    ECSStatus,
    EvidenceConflictResult,
    StanceCounts,
    StanceLabel,
    StancePrediction,
    StanceProbabilities,
)
from src.evidence_conflict.errors import StanceInputError


def _probabilities(
    supports: float = 0.8,
    refutes: float = 0.1,
    not_enough_info: float = 0.1,
) -> StanceProbabilities:
    return StanceProbabilities(supports, refutes, not_enough_info)


def _prediction(
    *,
    evidence_index: int = 0,
    label: StanceLabel = StanceLabel.SUPPORTS,
    confidence: float = 0.8,
    probabilities: StanceProbabilities | None = None,
    truncated: bool = False,
    abstained: bool = False,
) -> StancePrediction:
    return StancePrediction(
        evidence_index=evidence_index,
        label=label,
        confidence=confidence,
        probabilities=probabilities or _probabilities(),
        truncated=truncated,
        abstained=abstained,
    )


def _result(
    *,
    predictions: tuple[StancePrediction, ...] | None = None,
    counts: StanceCounts | None = None,
    status: ECSStatus = ECSStatus.COMPUTED,
    ecs: float | None = 0.5,
    **kwargs: object,
) -> EvidenceConflictResult:
    if predictions is None:
        predictions = (_prediction(),)
    if counts is None:
        counts = StanceCounts(supports=1)
    return EvidenceConflictResult(
        claim="A claim",
        predictions=predictions,
        counts=counts,
        status=status,
        ecs=ecs,
        **kwargs,
    )


def test_stance_labels_have_stable_string_values() -> None:
    assert StanceLabel.SUPPORTS.value == "supports"
    assert StanceLabel.REFUTES.value == "refutes"
    assert StanceLabel.NOT_ENOUGH_INFO.value == "not_enough_info"
    assert isinstance(StanceLabel.SUPPORTS, str)
    assert set(StanceLabel) == {
        StanceLabel.SUPPORTS,
        StanceLabel.REFUTES,
        StanceLabel.NOT_ENOUGH_INFO,
    }


def test_invalid_stance_label_value_is_rejected() -> None:
    with pytest.raises(ValueError):
        StanceLabel("support")


def test_prediction_rejects_string_label() -> None:
    with pytest.raises(StanceInputError):
        StancePrediction(
            evidence_index=0,
            label="supports",  # type: ignore[arg-type]
            confidence=0.8,
            probabilities=_probabilities(),
        )


def test_probabilities_accept_valid_values_and_coerce_integers() -> None:
    probabilities = StanceProbabilities(1, 0, 0)

    assert probabilities.supports == 1.0
    assert probabilities.refutes == 0.0
    assert probabilities.not_enough_info == 0.0


@pytest.mark.parametrize(
    ("supports", "refutes", "not_enough_info"),
    [
        (-0.1, 0.6, 0.5),
        (0.6, -0.1, 0.5),
        (0.6, 0.5, -0.1),
    ],
)
def test_probabilities_reject_out_of_range_values(
    supports: float,
    refutes: float,
    not_enough_info: float,
) -> None:
    with pytest.raises(StanceInputError):
        StanceProbabilities(supports, refutes, not_enough_info)


@pytest.mark.parametrize(
    ("supports", "refutes", "not_enough_info"),
    [
        (math.nan, 0.5, 0.5),
        (math.inf, 0.0, 0.0),
        (0.5, math.inf, -math.inf),
    ],
)
def test_probabilities_reject_non_finite_values(
    supports: float,
    refutes: float,
    not_enough_info: float,
) -> None:
    with pytest.raises(StanceInputError):
        StanceProbabilities(supports, refutes, not_enough_info)


@pytest.mark.parametrize(
    ("supports", "refutes", "not_enough_info"),
    [
        ("0.8", 0.1, 0.1),
        (True, 0.0, 0.0),
        (None, 0.5, 0.5),
    ],
)
def test_probabilities_reject_non_numeric_values(
    supports: object,
    refutes: object,
    not_enough_info: object,
) -> None:
    with pytest.raises(StanceInputError):
        StanceProbabilities(  # type: ignore[arg-type]
            supports,
            refutes,
            not_enough_info,
        )


def test_probabilities_reject_sum_that_is_not_one() -> None:
    with pytest.raises(StanceInputError, match="sum"):
        StanceProbabilities(0.5, 0.2, 0.1)


def test_probabilities_allow_small_sum_rounding_error() -> None:
    probabilities = StanceProbabilities(0.333333, 0.333333, 0.333334)

    assert probabilities.supports == pytest.approx(0.333333)


def test_prediction_defaults_and_flags() -> None:
    prediction = _prediction(truncated=True, abstained=True)

    assert prediction.evidence_index == 0
    assert prediction.label is StanceLabel.SUPPORTS
    assert prediction.confidence == 0.8
    assert prediction.truncated is True
    assert prediction.abstained is True


@pytest.mark.parametrize("index", [-1, 1.0, True, "0"])
def test_prediction_rejects_invalid_evidence_index(index: object) -> None:
    with pytest.raises(StanceInputError):
        StancePrediction(  # type: ignore[arg-type]
            evidence_index=index,
            label=StanceLabel.SUPPORTS,
            confidence=0.8,
            probabilities=_probabilities(),
        )


@pytest.mark.parametrize(
    "confidence",
    [-0.1, 1.1, math.nan, math.inf, "0.5", True],
)
def test_prediction_rejects_invalid_confidence(confidence: object) -> None:
    with pytest.raises(StanceInputError):
        StancePrediction(  # type: ignore[arg-type]
            evidence_index=0,
            label=StanceLabel.SUPPORTS,
            confidence=confidence,
            probabilities=_probabilities(),
        )


def test_prediction_rejects_invalid_probabilities() -> None:
    with pytest.raises(StanceInputError):
        StancePrediction(
            evidence_index=0,
            label=StanceLabel.SUPPORTS,
            confidence=0.8,
            probabilities=None,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("truncated", 1),
        ("abstained", "false"),
    ],
)
def test_prediction_rejects_non_boolean_flags(
    field: str,
    value: object,
) -> None:
    kwargs = {field: value}
    with pytest.raises(StanceInputError):
        StancePrediction(
            evidence_index=0,
            label=StanceLabel.SUPPORTS,
            confidence=0.8,
            probabilities=_probabilities(),
            **kwargs,
        )


def test_stance_counts_defaults_and_total() -> None:
    empty = StanceCounts()
    counts = StanceCounts(supports=2, refutes=3, not_enough_info=4)

    assert empty.total == 0
    assert counts.total == 9


@pytest.mark.parametrize(
    ("supports", "refutes", "not_enough_info"),
    [
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (1.0, 0, 0),
        (True, 0, 0),
        ("1", 0, 0),
    ],
)
def test_stance_counts_reject_invalid_values(
    supports: object,
    refutes: object,
    not_enough_info: object,
) -> None:
    with pytest.raises(StanceInputError):
        StanceCounts(  # type: ignore[arg-type]
            supports=supports,
            refutes=refutes,
            not_enough_info=not_enough_info,
        )


def test_ecs_status_values() -> None:
    assert {status.name: status.value for status in ECSStatus} == {
        "DISABLED": "disabled",
        "NO_EVIDENCE": "no_evidence",
        "RETRIEVAL_ERROR": "retrieval_error",
        "NLI_ERROR": "nli_error",
        "INSUFFICIENT_STANCE": "insufficient_stance",
        "COMPUTED": "computed",
    }


def test_result_accepts_computed_optional_values() -> None:
    predictions = (
        _prediction(),
        _prediction(
            evidence_index=1,
            label=StanceLabel.REFUTES,
            confidence=0.7,
            probabilities=StanceProbabilities(0.1, 0.8, 0.1),
        ),
    )
    result = _result(
        predictions=predictions,
        counts=StanceCounts(supports=1, refutes=1),
        ecs=0.0,
        evidence_coverage=1.0,
        refutation_share=0.5,
        support_share=0.5,
        mean_confidence=0.75,
    )

    assert result.claim == "A claim"
    assert result.predictions == predictions
    assert result.counts == StanceCounts(supports=1, refutes=1)
    assert result.status is ECSStatus.COMPUTED
    assert result.ecs == 0.0
    assert result.evidence_coverage == 1.0
    assert result.refutation_share == 0.5
    assert result.support_share == 0.5
    assert result.mean_confidence == 0.75


def test_result_accepts_no_evidence_with_none_metrics() -> None:
    result = _result(
        predictions=(),
        counts=StanceCounts(),
        status=ECSStatus.NO_EVIDENCE,
        ecs=None,
    )

    assert result.predictions == ()
    assert result.ecs is None
    assert result.evidence_coverage is None
    assert result.refutation_share is None
    assert result.support_share is None
    assert result.mean_confidence is None


def test_result_requires_ecs_when_computed() -> None:
    with pytest.raises(StanceInputError, match="COMPUTED"):
        _result(ecs=None)


def test_result_rejects_ecs_for_non_computed_status() -> None:
    with pytest.raises(StanceInputError, match="None"):
        _result(
            predictions=(),
            counts=StanceCounts(),
            status=ECSStatus.NO_EVIDENCE,
            ecs=0.0,
        )


def test_result_rejects_counts_that_do_not_match_predictions() -> None:
    with pytest.raises(StanceInputError, match="counts"):
        _result(counts=StanceCounts(supports=2))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ecs", math.nan),
        ("ecs", math.inf),
        ("ecs", "0.5"),
        ("evidence_coverage", -0.1),
        ("evidence_coverage", 1.1),
        ("refutation_share", math.nan),
        ("support_share", math.inf),
        ("mean_confidence", "0.5"),
    ],
)
def test_result_rejects_invalid_optional_metrics(
    field: str,
    value: object,
) -> None:
    kwargs = {field: value}
    with pytest.raises(StanceInputError):
        _result(**kwargs)


@pytest.mark.parametrize("claim", ["", "   ", None, 123])
def test_result_rejects_invalid_claim(claim: object) -> None:
    with pytest.raises(StanceInputError):
        EvidenceConflictResult(
            claim=claim,  # type: ignore[arg-type]
            predictions=(),
            counts=StanceCounts(),
            status=ECSStatus.NO_EVIDENCE,
        )


def test_result_rejects_non_prediction_records() -> None:
    with pytest.raises(StanceInputError, match="StancePrediction"):
        EvidenceConflictResult(
            claim="A claim",
            predictions=("not a prediction",),  # type: ignore[arg-type]
            counts=StanceCounts(),
            status=ECSStatus.NO_EVIDENCE,
        )


def test_result_rejects_invalid_status() -> None:
    with pytest.raises(StanceInputError, match="ECSStatus"):
        EvidenceConflictResult(
            claim="A claim",
            predictions=(),
            counts=StanceCounts(),
            status="computed",  # type: ignore[arg-type]
            ecs=0.0,
        )


def test_result_normalizes_prediction_sequence_to_tuple() -> None:
    prediction = _prediction()
    result = EvidenceConflictResult(
        claim="A claim",
        predictions=[prediction],  # type: ignore[arg-type]
        counts=StanceCounts(supports=1),
        status=ECSStatus.COMPUTED,
        ecs=0.5,
    )

    assert result.predictions == (prediction,)
