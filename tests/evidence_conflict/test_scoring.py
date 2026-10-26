"""Offline unit tests for pure Evidence Conflict Score aggregation.

These tests use manually constructed stance predictions. They do not load an
NLI model, download weights, or call external services.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from src.evidence_conflict import (
    ECSStatus,
    StanceCounts,
    StanceLabel,
    StancePrediction,
    StanceProbabilities,
)
from src.evidence_conflict.errors import StanceInputError
from src.evidence_conflict.scoring import compute_evidence_conflict

CLAIM = "The policy reduced emissions."
SUPPORTS = StanceLabel.SUPPORTS
REFUTES = StanceLabel.REFUTES
NOT_ENOUGH_INFO = StanceLabel.NOT_ENOUGH_INFO


def _probabilities(label: StanceLabel) -> StanceProbabilities:
    if label is SUPPORTS:
        return StanceProbabilities(0.8, 0.1, 0.1)
    if label is REFUTES:
        return StanceProbabilities(0.1, 0.8, 0.1)
    return StanceProbabilities(0.1, 0.1, 0.8)


def _prediction(
    evidence_index: int,
    label: StanceLabel,
    confidence: float = 0.8,
    *,
    truncated: bool = False,
    abstained: bool = False,
) -> StancePrediction:
    return StancePrediction(
        evidence_index=evidence_index,
        label=label,
        confidence=confidence,
        probabilities=_probabilities(label),
        truncated=truncated,
        abstained=abstained,
    )


def _predictions(
    labels: Sequence[StanceLabel],
    confidences: Sequence[float] | None = None,
) -> list[StancePrediction]:
    if confidences is None:
        confidences = [0.8] * len(labels)
    assert len(confidences) == len(labels)
    return [
        _prediction(index, label, confidence)
        for index, (label, confidence) in enumerate(zip(labels, confidences))
    ]


def _counts(
    supports: int = 0,
    refutes: int = 0,
    not_enough_info: int = 0,
) -> StanceCounts:
    return StanceCounts(
        supports=supports,
        refutes=refutes,
        not_enough_info=not_enough_info,
    )


def test_empty_predictions_return_no_evidence() -> None:
    result = compute_evidence_conflict([], claim_text=CLAIM)

    assert result.claim == CLAIM
    assert result.predictions == ()
    assert result.counts == _counts()
    assert result.counts.total == 0
    assert result.status is ECSStatus.NO_EVIDENCE
    assert result.ecs is None
    assert result.evidence_coverage is None
    assert result.support_share is None
    assert result.refutation_share is None
    assert result.mean_confidence is None


def test_all_supports_have_zero_ecs_and_full_support_share() -> None:
    predictions = _predictions(
        [SUPPORTS, SUPPORTS, SUPPORTS],
        [0.9, 0.8, 0.7],
    )

    result = compute_evidence_conflict(predictions, claim_text=CLAIM)

    assert result.predictions == tuple(predictions)
    assert result.counts == _counts(supports=3)
    assert result.status is ECSStatus.COMPUTED
    assert result.ecs == 0.0
    assert result.evidence_coverage == 1.0
    assert result.support_share == 1.0
    assert result.refutation_share == 0.0
    assert result.mean_confidence == pytest.approx(0.8)


def test_all_refutes_have_zero_ecs_and_full_refutation_share() -> None:
    predictions = _predictions([REFUTES, REFUTES], [0.7, 0.6])

    result = compute_evidence_conflict(predictions, claim_text=CLAIM)

    assert result.counts == _counts(refutes=2)
    assert result.status is ECSStatus.COMPUTED
    assert result.ecs == 0.0
    assert result.evidence_coverage == 1.0
    assert result.support_share == 0.0
    assert result.refutation_share == 1.0
    assert result.mean_confidence == pytest.approx(0.65)


def test_balanced_support_and_refute_evidence_has_maximum_ecs() -> None:
    predictions = _predictions(
        [SUPPORTS, SUPPORTS, REFUTES, REFUTES],
        [0.9, 0.8, 0.7, 0.6],
    )

    result = compute_evidence_conflict(predictions, claim_text=CLAIM)

    assert result.counts == _counts(supports=2, refutes=2)
    assert result.status is ECSStatus.COMPUTED
    assert result.ecs == 1.0
    assert result.evidence_coverage == 1.0
    assert result.support_share == 0.5
    assert result.refutation_share == 0.5
    assert result.mean_confidence == pytest.approx(0.75)


def test_balanced_evidence_has_higher_ecs_than_imbalanced_evidence() -> None:
    balanced = compute_evidence_conflict(
        _predictions([SUPPORTS, SUPPORTS, REFUTES, REFUTES]),
        claim_text=CLAIM,
    )
    imbalanced = compute_evidence_conflict(
        _predictions([SUPPORTS, SUPPORTS, SUPPORTS, REFUTES]),
        claim_text=CLAIM,
    )

    assert balanced.ecs > imbalanced.ecs


def test_imbalanced_support_and_refute_counts_use_the_minority_stance() -> None:
    predictions = _predictions(
        [SUPPORTS, SUPPORTS, SUPPORTS, REFUTES],
        [0.9, 0.8, 0.7, 0.6],
    )

    result = compute_evidence_conflict(predictions, claim_text=CLAIM)

    assert result.counts == _counts(supports=3, refutes=1)
    assert result.status is ECSStatus.COMPUTED
    assert result.ecs == pytest.approx(0.5)
    assert result.evidence_coverage == 1.0
    assert result.support_share == pytest.approx(0.75)
    assert result.refutation_share == pytest.approx(0.25)


def test_mixed_support_refute_and_nei_predictions() -> None:
    predictions = _predictions(
        [
            SUPPORTS,
            SUPPORTS,
            REFUTES,
            REFUTES,
            NOT_ENOUGH_INFO,
            NOT_ENOUGH_INFO,
        ],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
    )

    result = compute_evidence_conflict(predictions, claim_text=CLAIM)

    assert result.counts == _counts(
        supports=2,
        refutes=2,
        not_enough_info=2,
    )
    assert result.status is ECSStatus.COMPUTED
    assert result.ecs == pytest.approx(2.0 * 2.0 / 6.0)
    assert result.evidence_coverage == pytest.approx(4.0 / 6.0)
    assert result.support_share == pytest.approx(0.5)
    assert result.refutation_share == pytest.approx(0.5)
    assert result.mean_confidence == pytest.approx(0.65)


def test_not_enough_info_reduces_coverage_adjusted_ecs() -> None:
    balanced = compute_evidence_conflict(
        _predictions([SUPPORTS, SUPPORTS, REFUTES, REFUTES]),
        claim_text=CLAIM,
    )
    with_nei = compute_evidence_conflict(
        _predictions(
            [
                SUPPORTS,
                SUPPORTS,
                REFUTES,
                REFUTES,
                NOT_ENOUGH_INFO,
                NOT_ENOUGH_INFO,
            ]
        ),
        claim_text=CLAIM,
    )

    assert balanced.ecs == 1.0
    assert with_nei.ecs == pytest.approx(2.0 / 3.0)
    assert with_nei.evidence_coverage == pytest.approx(2.0 / 3.0)
    assert with_nei.ecs < balanced.ecs


def test_all_not_enough_info_is_insufficient_stance() -> None:
    predictions = _predictions(
        [NOT_ENOUGH_INFO, NOT_ENOUGH_INFO],
        [0.6, 0.4],
    )

    result = compute_evidence_conflict(predictions, claim_text=CLAIM)

    assert result.counts == _counts(not_enough_info=2)
    assert result.status is ECSStatus.INSUFFICIENT_STANCE
    assert result.ecs is None
    assert result.evidence_coverage == 0.0
    assert result.support_share is None
    assert result.refutation_share is None
    assert result.mean_confidence == pytest.approx(0.5)


@pytest.mark.parametrize(
    (
        "label",
        "expected_status",
        "expected_ecs",
        "expected_coverage",
        "expected_support_share",
        "expected_refutation_share",
    ),
    [
        (SUPPORTS, ECSStatus.COMPUTED, 0.0, 1.0, 1.0, 0.0),
        (REFUTES, ECSStatus.COMPUTED, 0.0, 1.0, 0.0, 1.0),
        (
            NOT_ENOUGH_INFO,
            ECSStatus.INSUFFICIENT_STANCE,
            None,
            0.0,
            None,
            None,
        ),
    ],
)
def test_single_prediction_status_and_metrics(
    label: StanceLabel,
    expected_status: ECSStatus,
    expected_ecs: float | None,
    expected_coverage: float,
    expected_support_share: float | None,
    expected_refutation_share: float | None,
) -> None:
    prediction = _prediction(0, label, 0.7)

    result = compute_evidence_conflict([prediction], claim_text=CLAIM)

    assert result.counts == _counts(
        supports=int(label is SUPPORTS),
        refutes=int(label is REFUTES),
        not_enough_info=int(label is NOT_ENOUGH_INFO),
    )
    assert result.status is expected_status
    assert result.ecs == expected_ecs
    assert result.evidence_coverage == expected_coverage
    assert result.support_share == expected_support_share
    assert result.refutation_share == expected_refutation_share
    assert result.mean_confidence == pytest.approx(0.7)


def test_swapping_supports_and_refutes_preserves_ecs_but_swaps_shares() -> None:
    confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
    support_heavy = compute_evidence_conflict(
        _predictions(
            [SUPPORTS, SUPPORTS, SUPPORTS, REFUTES, NOT_ENOUGH_INFO],
            confidences,
        ),
        claim_text=CLAIM,
    )
    refute_heavy = compute_evidence_conflict(
        _predictions(
            [REFUTES, REFUTES, REFUTES, SUPPORTS, NOT_ENOUGH_INFO],
            confidences,
        ),
        claim_text=CLAIM,
    )

    assert support_heavy.ecs == refute_heavy.ecs == pytest.approx(0.4)
    assert support_heavy.evidence_coverage == refute_heavy.evidence_coverage == 0.8
    assert support_heavy.support_share == pytest.approx(0.75)
    assert refute_heavy.support_share == pytest.approx(0.25)
    assert support_heavy.refutation_share == pytest.approx(0.25)
    assert refute_heavy.refutation_share == pytest.approx(0.75)
    assert (
        support_heavy.mean_confidence
        == refute_heavy.mean_confidence
        == pytest.approx(0.7)
    )


def test_ecs_and_proportions_stay_in_unit_interval() -> None:
    for supports in range(6):
        for refutes in range(6):
            for not_enough_info in range(3):
                n_total = supports + refutes + not_enough_info
                if n_total == 0:
                    continue

                labels = (
                    [SUPPORTS] * supports
                    + [REFUTES] * refutes
                    + [NOT_ENOUGH_INFO] * not_enough_info
                )
                result = compute_evidence_conflict(
                    _predictions(labels),
                    claim_text=CLAIM,
                )

                assert result.counts == _counts(
                    supports=supports,
                    refutes=refutes,
                    not_enough_info=not_enough_info,
                )
                assert result.counts.total == n_total
                assert result.evidence_coverage is not None
                assert 0.0 <= result.evidence_coverage <= 1.0
                assert result.evidence_coverage == pytest.approx(
                    (supports + refutes) / n_total
                )

                n_stance = supports + refutes
                if n_stance == 0:
                    assert result.status is ECSStatus.INSUFFICIENT_STANCE
                    assert result.ecs is None
                    assert result.support_share is None
                    assert result.refutation_share is None
                else:
                    assert result.status is ECSStatus.COMPUTED
                    assert result.ecs is not None
                    assert 0.0 <= result.ecs <= 1.0
                    assert result.ecs == pytest.approx(
                        2.0 * min(supports, refutes) / n_total
                    )
                    assert result.support_share is not None
                    assert result.refutation_share is not None
                    assert 0.0 <= result.support_share <= 1.0
                    assert 0.0 <= result.refutation_share <= 1.0
                    assert result.support_share == pytest.approx(
                        supports / n_stance
                    )
                    assert result.refutation_share == pytest.approx(
                        refutes / n_stance
                    )
                    assert (
                        result.support_share + result.refutation_share
                        == pytest.approx(1.0)
                    )


def test_coverage_and_mean_confidence_use_all_predictions() -> None:
    predictions = _predictions(
        [SUPPORTS, REFUTES, NOT_ENOUGH_INFO],
        [0.9, 0.6, 0.3],
    )

    result = compute_evidence_conflict(predictions, claim_text=CLAIM)

    assert result.evidence_coverage == pytest.approx(2.0 / 3.0)
    assert result.mean_confidence == pytest.approx(0.6)
    assert result.ecs == pytest.approx(2.0 / 3.0)


def test_confidence_does_not_change_ecs_or_shares() -> None:
    labels = [SUPPORTS, REFUTES, NOT_ENOUGH_INFO]
    low_confidence = compute_evidence_conflict(
        _predictions(labels, [0.1, 0.2, 0.3]),
        claim_text=CLAIM,
    )
    high_confidence = compute_evidence_conflict(
        _predictions(labels, [0.9, 0.8, 0.7]),
        claim_text=CLAIM,
    )

    for field in (
        "ecs",
        "evidence_coverage",
        "support_share",
        "refutation_share",
    ):
        assert getattr(low_confidence, field) == pytest.approx(
            getattr(high_confidence, field)
        )

    assert low_confidence.mean_confidence == pytest.approx(0.2)
    assert high_confidence.mean_confidence == pytest.approx(0.8)


@pytest.mark.parametrize(
    "predictions",
    [
        None,
        42,
        object(),
        "not predictions",
        b"not predictions",
        bytearray(b"not predictions"),
        {},
        {"supports": 1},
        _prediction(0, SUPPORTS),
    ],
)
def test_invalid_prediction_collections_are_rejected(
    predictions: object,
) -> None:
    with pytest.raises(StanceInputError):
        compute_evidence_conflict(
            predictions,  # type: ignore[arg-type]
            claim_text=CLAIM,
        )


@pytest.mark.parametrize(
    "item",
    [
        object(),
        None,
        1,
        "supports",
        {"label": SUPPORTS},
    ],
)
def test_malformed_prediction_items_are_rejected(item: object) -> None:
    with pytest.raises(
        StanceInputError,
        match=r"predictions\[0\] must be a StancePrediction",
    ):
        compute_evidence_conflict(
            [item],  # type: ignore[list-item]
            claim_text=CLAIM,
        )


def test_invalid_stance_label_is_rejected() -> None:
    prediction = _prediction(0, SUPPORTS)
    object.__setattr__(prediction, "label", "supports")

    with pytest.raises(StanceInputError, match="label must be a StanceLabel"):
        compute_evidence_conflict([prediction], claim_text=CLAIM)


@pytest.mark.parametrize("claim_text", [None, 123, "", "   ", "\n\t"])
def test_invalid_claim_text_is_rejected(claim_text: object) -> None:
    with pytest.raises(StanceInputError, match="claim_text"):
        compute_evidence_conflict(
            [],
            claim_text=claim_text,  # type: ignore[arg-type]
        )


def test_claim_text_is_preserved_unchanged() -> None:
    claim = "  A claim with surrounding whitespace  "

    result = compute_evidence_conflict([], claim_text=claim)

    assert result.claim == claim


def test_prediction_order_is_preserved() -> None:
    predictions = [
        _prediction(2, REFUTES, 0.6),
        _prediction(0, SUPPORTS, 0.9),
        _prediction(1, NOT_ENOUGH_INFO, 0.4),
    ]

    result = compute_evidence_conflict(predictions, claim_text=CLAIM)

    assert result.predictions == tuple(predictions)
    assert [prediction.evidence_index for prediction in result.predictions] == [
        2,
        0,
        1,
    ]
    assert [prediction.label for prediction in result.predictions] == [
        REFUTES,
        SUPPORTS,
        NOT_ENOUGH_INFO,
    ]
    assert result.counts == _counts(
        supports=1,
        refutes=1,
        not_enough_info=1,
    )


def test_generator_predictions_are_supported() -> None:
    predictions = (
        _prediction(index, label)
        for index, label in enumerate(
            [SUPPORTS, REFUTES, NOT_ENOUGH_INFO]
        )
    )

    result = compute_evidence_conflict(predictions, claim_text=CLAIM)

    assert result.counts == _counts(
        supports=1,
        refutes=1,
        not_enough_info=1,
    )
    assert [prediction.evidence_index for prediction in result.predictions] == [
        0,
        1,
        2,
    ]


def test_inputs_are_not_mutated() -> None:
    predictions = [
        _prediction(0, SUPPORTS, 0.9),
        _prediction(1, REFUTES, 0.6),
        _prediction(2, NOT_ENOUGH_INFO, 0.3),
    ]
    original = list(predictions)
    original_objects = tuple(predictions)
    original_reprs = tuple(repr(prediction) for prediction in predictions)

    result = compute_evidence_conflict(predictions, claim_text=CLAIM)

    assert predictions == original
    assert tuple(predictions) == original_objects
    assert all(
        actual is expected
        for actual, expected in zip(predictions, original_objects)
    )
    assert tuple(repr(prediction) for prediction in predictions) == original_reprs
    assert result.predictions == original_objects
    assert all(
        actual is expected
        for actual, expected in zip(result.predictions, original_objects)
    )


@pytest.mark.parametrize(
    ("labels", "expected_status", "expected_counts"),
    [
        ([], ECSStatus.NO_EVIDENCE, _counts()),
        (
            [NOT_ENOUGH_INFO],
            ECSStatus.INSUFFICIENT_STANCE,
            _counts(not_enough_info=1),
        ),
        ([SUPPORTS], ECSStatus.COMPUTED, _counts(supports=1)),
        ([REFUTES], ECSStatus.COMPUTED, _counts(refutes=1)),
        (
            [SUPPORTS, REFUTES],
            ECSStatus.COMPUTED,
            _counts(supports=1, refutes=1),
        ),
        (
            [SUPPORTS, NOT_ENOUGH_INFO],
            ECSStatus.COMPUTED,
            _counts(supports=1, not_enough_info=1),
        ),
        (
            [REFUTES, NOT_ENOUGH_INFO],
            ECSStatus.COMPUTED,
            _counts(refutes=1, not_enough_info=1),
        ),
        (
            [SUPPORTS, REFUTES, NOT_ENOUGH_INFO],
            ECSStatus.COMPUTED,
            _counts(1, 1, 1),
        ),
    ],
)
def test_status_and_counts_are_consistent(
    labels: list[StanceLabel],
    expected_status: ECSStatus,
    expected_counts: StanceCounts,
) -> None:
    result = compute_evidence_conflict(
        _predictions(labels),
        claim_text=CLAIM,
    )

    assert result.status is expected_status
    assert result.counts == expected_counts
    assert result.counts.total == len(result.predictions)
    assert result.counts.supports == sum(
        prediction.label is SUPPORTS for prediction in result.predictions
    )
    assert result.counts.refutes == sum(
        prediction.label is REFUTES for prediction in result.predictions
    )
    assert result.counts.not_enough_info == sum(
        prediction.label is NOT_ENOUGH_INFO
        for prediction in result.predictions
    )
    if expected_status is ECSStatus.COMPUTED:
        assert result.ecs is not None
    else:
        assert result.ecs is None


def test_undefined_ecs_is_none_not_zero() -> None:
    empty = compute_evidence_conflict([], claim_text=CLAIM)
    all_nei = compute_evidence_conflict(
        _predictions([NOT_ENOUGH_INFO]),
        claim_text=CLAIM,
    )

    assert empty.ecs is None
    assert all_nei.ecs is None
    assert empty.ecs != 0.0
    assert all_nei.ecs != 0.0


def test_abstained_and_truncated_predictions_are_still_counted() -> None:
    predictions = [
        _prediction(0, SUPPORTS, 0.9, truncated=True),
        _prediction(1, NOT_ENOUGH_INFO, 0.4, abstained=True),
    ]

    result = compute_evidence_conflict(predictions, claim_text=CLAIM)

    assert result.counts == _counts(
        supports=1,
        not_enough_info=1,
    )
    assert result.status is ECSStatus.COMPUTED
    assert result.ecs == 0.0
    assert result.evidence_coverage == 0.5
    assert result.support_share == 1.0
    assert result.refutation_share == 0.0
    assert result.mean_confidence == pytest.approx(0.65)
