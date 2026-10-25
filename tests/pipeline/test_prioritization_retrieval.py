"""Tests for the prioritization-to-evidence-retrieval pipeline."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from src.pipeline import prioritize_and_retrieve
from src.prioritization import (
    FEATURE_COLUMNS,
    Claim,
    Objective,
    UserRating,
)
from src.retrieval import EvidenceSearchError


class IdentityScaler:
    n_features_in_ = len(FEATURE_COLUMNS)
    feature_names_in_ = np.asarray(FEATURE_COLUMNS, dtype=object)

    def transform(self, values: object) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)


class FakeModel:
    def __init__(self, label: str, classes: tuple[str, ...]) -> None:
        self.label = label
        self.classes_ = np.asarray(classes, dtype=object)

    def predict(self, values: object) -> np.ndarray:
        return np.asarray([self.label] * len(np.asarray(values)), dtype=object)

    def predict_proba(self, values: object) -> np.ndarray:
        return np.asarray(
            [[0.9 if value == self.label else 0.1 for value in self.classes_]
             for _ in np.asarray(values)]
        )


class FakeStore:
    def __init__(self, o1_model: FakeModel | None = None, o2_model: FakeModel | None = None):
        self.scaler = IdentityScaler()
        self.o1_model = o1_model
        self.o2_model = o2_model


def _claims() -> list[Claim]:
    return [Claim("a", "Claim A"), Claim("b", "Claim B"), Claim("c", "Claim C")]


def _ratings() -> list[UserRating]:
    return [
        UserRating("a", "True"),
        UserRating("b", "False"),
        UserRating("c", "Mixture"),
    ]


def test_prioritization_order_and_metadata_are_preserved() -> None:
    retrieval = Mock(
        side_effect=lambda claim, **kwargs: [
            {"title": claim, "url": f"https://example.test/{claim[-1]}", "content": claim}
        ]
    )

    results = prioritize_and_retrieve(
        _claims(),
        _ratings(),
        "O1",
        artifact_store=FakeStore(FakeModel("True", ("False", "True"))),
        retrieve=retrieval,
    )

    assert [result.claim_id for result in results] == ["a", "b", "c"]
    assert [result.initial_rank for result in results] == [1, 2, 3]
    assert [result.claim for result in results] == ["Claim A", "Claim B", "Claim C"]
    assert all(result.objective is Objective.O1 for result in results)
    assert all(result.predicted_truth == "True" for result in results)
    assert [call.args[0] for call in retrieval.call_args_list] == [
        "Claim A", "Claim B", "Claim C"
    ]
    assert retrieval.call_count == 3
    assert results[1].evidence[0]["url"] == "https://example.test/B"


def test_o1_true_and_o2_low_claims_are_not_filtered() -> None:
    retrieval = Mock(return_value=[])
    o1 = prioritize_and_retrieve(
        _claims(),
        _ratings(),
        Objective.O1,
        artifact_store=FakeStore(FakeModel("True", ("False", "True"))),
        retrieve=retrieval,
    )
    retrieval.reset_mock()
    o2 = prioritize_and_retrieve(
        _claims(),
        _ratings(),
        Objective.O2,
        artifact_store=FakeStore(o2_model=FakeModel("Low", ("Low", "High"))),
        retrieve=retrieval,
    )

    assert len(o1) == len(_claims())
    assert all(result.predicted_truth == "True" for result in o1)
    assert len(o2) == len(_claims())
    assert all(result.predicted_tpb == "Low" for result in o2)
    assert retrieval.call_count == len(_claims())


def test_o3_metadata_and_order_are_unchanged() -> None:
    claims = [Claim("a", "Claim A"), Claim("b", "Claim B"), Claim("c", "Claim C")]
    ratings = [
        UserRating("a", "True"), UserRating("a", "False"),
        UserRating("b", "True"), UserRating("b", "True"),
        UserRating("c", "Mixture"), UserRating("c", "Mostly True"),
    ]
    retrieval = Mock(return_value=[])

    results = prioritize_and_retrieve(
        claims, ratings, Objective.O3, artifact_store=FakeStore(), retrieve=retrieval
    )

    assert [result.claim_id for result in results] == ["a", "c", "b"]
    assert [result.initial_rank for result in results] == [1, 2, 3]
    assert [result.disputability for result in results] == pytest.approx(
        [2.0, 0.125, 0.0]
    )
    assert all(result.predicted_truth is None for result in results)
    assert all(result.predicted_tpb is None for result in results)


def test_empty_evidence_remains_a_successful_result() -> None:
    retrieval = Mock(return_value=[])

    results = prioritize_and_retrieve(
        _claims(), _ratings(), "O3", retrieve=retrieval
    )

    assert len(results) == 3
    assert all(result.evidence == [] for result in results)
    assert all(result.retrieval_error is None for result in results)


def test_expected_retrieval_error_is_recorded_and_processing_continues() -> None:
    def retrieve(claim: str, **kwargs: object) -> list[dict[str, object]]:
        if claim == "Claim B":
            raise EvidenceSearchError("search unavailable")
        return [{"url": "https://example.test/source", "content": claim}]

    results = prioritize_and_retrieve(
        _claims(), _ratings(), "O3", retrieve=retrieve
    )

    assert [result.claim_id for result in results] == ["a", "b", "c"]
    assert results[0].evidence[0]["content"] == "Claim A"
    assert results[1].evidence == []
    assert results[1].retrieval_error is not None
    assert results[1].retrieval_error.error_type == "EvidenceSearchError"
    assert results[1].retrieval_error.message == "search unavailable"
    assert results[2].evidence[0]["content"] == "Claim C"


def test_invalid_prioritization_input_uses_existing_validation() -> None:
    with pytest.raises(ValueError):
        prioritize_and_retrieve(
            [Claim("a", "Claim A")], [], "O3", retrieve=Mock()
        )
