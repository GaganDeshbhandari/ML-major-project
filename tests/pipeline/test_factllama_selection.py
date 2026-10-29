from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.factllama import Prediction
from src.evidence_conflict import (
    StanceLabel,
    StancePrediction,
    StanceProbabilities,
)
from src.pipeline import (
    FactLLaMASelectionError,
    FactLLaMASelectionService,
    PrioritizationRetrievalResult,
    fuse_prioritization_results,
    prioritize_and_retrieve,
)
from src.prioritization import FEATURE_COLUMNS, Claim, Objective, UserRating

import numpy as np


class _IdentityScaler:
    n_features_in_ = len(FEATURE_COLUMNS)
    feature_names_in_ = np.asarray(FEATURE_COLUMNS, dtype=object)

    def transform(self, values: object) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)


class _Model:
    def __init__(self, label: str, classes: tuple[str, ...]) -> None:
        self.label = label
        self.classes_ = np.asarray(classes, dtype=object)

    def predict(self, values: object) -> np.ndarray:
        return np.asarray([self.label] * len(np.asarray(values)), dtype=object)

    def predict_proba(self, values: object) -> np.ndarray:
        return np.asarray(
            [
                [
                    0.9 if value == self.label else 0.1
                    for value in self.classes_
                ]
                for _ in np.asarray(values)
            ]
        )


class _Store:
    def __init__(self) -> None:
        self.scaler = _IdentityScaler()
        self.o1_model = _Model("True", ("False", "True"))
        self.o2_model = None


class _StanceClassifier:
    def __init__(self, labels: dict[str, StanceLabel]) -> None:
        self.labels = labels
        self.calls: list[tuple[str, list[dict[str, object]]]] = []

    def predict(
        self,
        claim: str,
        evidence: list[dict[str, object]],
    ) -> list[StancePrediction]:
        self.calls.append((claim, evidence))
        label = self.labels[claim]
        probability = StanceProbabilities(
            0.8 if label is StanceLabel.SUPPORTS else 0.1,
            0.8 if label is StanceLabel.REFUTES else 0.1,
            0.1,
        )
        return [
            StancePrediction(
                evidence_index=index,
                label=label,
                confidence=0.8,
                probabilities=probability,
            )
            for index in range(len(evidence))
        ]


def _results() -> list[PrioritizationRetrievalResult]:
    return [
        PrioritizationRetrievalResult(
            claim_id="a",
            claim="Claim A",
            objective=Objective.O1,
            initial_rank=1,
            evidence=[{"title": "A source", "url": "https://a", "content": "A evidence"}],
        ),
        PrioritizationRetrievalResult(
            claim_id="b",
            claim="Claim B",
            objective=Objective.O1,
            initial_rank=2,
            evidence=[{"title": "B source", "url": "https://b", "content": "B evidence"}],
        ),
    ]


def test_selected_claim_invokes_factllama_once_with_only_selected_evidence() -> None:
    predictor = Mock()
    predictor.predict.return_value = Prediction(
        claim="Claim B",
        verdict="half",
        raw_output="half",
        evidence_used=3,
    )
    service = FactLLaMASelectionService(predictor=predictor)

    result = service.verify_selected_claim(_results(), "b")

    assert result.claim_id == "b"
    assert result.claim == "Claim B"
    assert result.verdict == "half"
    predictor.predict.assert_called_once_with(
        "Claim B",
        ({"title": "B source", "url": "https://b", "content": "B evidence"},),
    )


def test_invalid_selection_does_not_invoke_factllama() -> None:
    predictor = Mock()
    service = FactLLaMASelectionService(predictor=predictor)

    with pytest.raises(FactLLaMASelectionError, match="not found"):
        service.verify_selected_claim(_results(), "missing")

    predictor.predict.assert_not_called()


def test_missing_evidence_is_explicit_and_does_not_invoke_factllama() -> None:
    predictor = Mock()
    service = FactLLaMASelectionService(predictor=predictor)
    results = _results()
    results[0].evidence = []

    with pytest.raises(FactLLaMASelectionError, match="No evidence"):
        service.verify_selected_claim(results, "a")

    predictor.predict.assert_not_called()


def test_unparseable_output_is_returned_without_inventing_verdict() -> None:
    predictor = Mock()
    predictor.predict.return_value = Prediction(
        claim="Claim A",
        verdict=None,
        raw_output="uncertain",
    )
    service = FactLLaMASelectionService(predictor=predictor)

    result = service.verify_selected_claim(_results(), "a")

    assert result.verdict is None
    assert result.raw_output == "uncertain"
    assert result.error is not None
    assert result.error.error_type == "VerdictParseError"


def test_inference_error_is_explicit_and_does_not_affect_other_claims() -> None:
    predictor = Mock()
    predictor.predict.side_effect = RuntimeError("model unavailable")
    service = FactLLaMASelectionService(predictor=predictor)

    result = service.verify_selected_claim(_results(), "a")

    assert result.verdict is None
    assert result.error is not None
    assert result.error.error_type == "RuntimeError"
    assert result.error.message == "model unavailable"
    predictor.predict.assert_called_once()


def test_full_mocked_pipeline_fuses_results_then_verifies_only_selected_claim() -> None:
    claims = [
        Claim("a", "Claim A"),
        Claim("b", "Claim B"),
        Claim("c", "Claim C"),
    ]
    ratings = [
        UserRating("a", "True"),
        UserRating("a", "False"),
        UserRating("b", "True"),
        UserRating("b", "True"),
        UserRating("c", "True"),
        UserRating("c", "False"),
        UserRating("c", "Mixture"),
    ]
    evidence_by_claim = {
        claim.claim: [
            {
                "title": f"{claim.claim} source",
                "url": f"https://example.test/{claim.claim[-1]}",
                "content": f"{claim.claim} evidence",
                "relevance_score": 0.9,
                "rerank_score": 0.8,
            }
        ]
        for claim in claims
    }

    def retrieve(claim: str, **_: object) -> list[dict[str, object]]:
        return evidence_by_claim[claim]

    stance = _StanceClassifier(
        {
            "Claim A": StanceLabel.SUPPORTS,
            "Claim B": StanceLabel.REFUTES,
            "Claim C": StanceLabel.SUPPORTS,
        }
    )
    retrieved = prioritize_and_retrieve(
        claims,
        ratings,
        Objective.O3,
        retrieve=retrieve,
        stance_classifier=stance,
    )
    fused = fuse_prioritization_results(retrieved)

    assert [item.claim_id for item in fused] == ["a", "c", "b"]
    assert fused[0].result.evidence[0]["url"] == "https://example.test/A"
    assert fused[0].result.ecs_result is not None
    assert fused[0].result.ecs_result.status.value == "computed"
    assert [claim for claim, _ in stance.calls] == [
        "Claim A",
        "Claim C",
        "Claim B",
    ]

    predictor = Mock()
    predictor.predict.return_value = Prediction(
        claim="Claim C",
        verdict="true",
        raw_output="true",
    )
    service = FactLLaMASelectionService(predictor=predictor)
    verification = service.verify_selected_claim(
        [item.result for item in fused],
        "c",
    )

    assert verification.claim_id == "c"
    assert verification.claim == "Claim C"
    assert verification.evidence == tuple(evidence_by_claim["Claim C"])
    predictor.predict.assert_called_once_with(
        "Claim C",
        tuple(evidence_by_claim["Claim C"]),
    )


def test_full_flow_keeps_empty_and_failed_retrieval_unverifiable() -> None:
    claims = [Claim("empty", "No evidence"), Claim("failed", "Failed search")]
    ratings = [
        UserRating("empty", "True"),
        UserRating("empty", "True"),
        UserRating("failed", "False"),
        UserRating("failed", "False"),
    ]

    def retrieve(claim: str, **_: object) -> list[dict[str, object]]:
        if claim == "Failed search":
            from src.retrieval import EvidenceSearchError

            raise EvidenceSearchError("search unavailable")
        return []

    results = prioritize_and_retrieve(
        claims,
        ratings,
        Objective.O3,
        retrieve=retrieve,
    )
    predictor = Mock()
    service = FactLLaMASelectionService(predictor=predictor)

    for result in results:
        with pytest.raises(FactLLaMASelectionError):
            service.verify_selected_claim(results, result.claim_id)

    predictor.predict.assert_not_called()
