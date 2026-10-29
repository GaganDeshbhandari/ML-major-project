"""Offline tests for the end-to-end ClaimRank service."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from src.evidence_conflict import (
    ECSStatus,
    StanceLabel,
    StancePrediction,
    StanceProbabilities,
)
from src.evidence_conflict.errors import StanceInferenceError
from src.pipeline import (
    ClaimRankObjective,
    ClaimRankService,
    ClaimRankServiceError,
)
from src.prioritization import (
    FEATURE_COLUMNS,
    ArtifactPaths,
    ArtifactStore,
    Claim,
    Objective,
    UserRating,
)
from src.retrieval import EvidenceSearchError


class FakeModel:
    def __init__(self, label: str, classes: tuple[str, ...]) -> None:
        self.label = label
        self.classes_ = np.asarray(classes, dtype=object)

    def predict(self, values: object) -> np.ndarray:
        return np.asarray([self.label] * len(np.asarray(values)), dtype=object)

    def predict_proba(self, values: object) -> np.ndarray:
        return np.asarray(
            [
                [0.9 if value == self.label else 0.1 for value in self.classes_]
                for _ in np.asarray(values)
            ]
        )


class IdentityScaler:
    n_features_in_ = len(FEATURE_COLUMNS)
    feature_names_in_ = np.asarray(FEATURE_COLUMNS, dtype=object)

    def transform(self, values: object) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)


class FakeStore:
    def __init__(
        self,
        o1_model: FakeModel | None = None,
        o2_model: FakeModel | None = None,
    ) -> None:
        self.scaler = IdentityScaler()
        self.o1_model = o1_model
        self.o2_model = o2_model


class FakeStanceClassifier:
    """Deterministic stance labels that exercise the real ECS scorer."""

    _LABELS = {
        "Claim A": [StanceLabel.SUPPORTS] * 2 + [StanceLabel.NOT_ENOUGH_INFO] * 2,
        "Claim B": [StanceLabel.SUPPORTS] * 2 + [StanceLabel.REFUTES] * 2,
        "Claim C": [StanceLabel.SUPPORTS] + [StanceLabel.REFUTES] * 3,
    }

    def predict(self, claim: str, evidence: list[dict]) -> list[StancePrediction]:
        labels = self._LABELS[claim]
        assert len(labels) == len(evidence)
        probabilities = {
            StanceLabel.SUPPORTS: StanceProbabilities(0.9, 0.05, 0.05),
            StanceLabel.REFUTES: StanceProbabilities(0.05, 0.9, 0.05),
            StanceLabel.NOT_ENOUGH_INFO: StanceProbabilities(0.1, 0.1, 0.8),
        }
        return [
            StancePrediction(
                evidence_index=index,
                label=label,
                confidence=0.9,
                probabilities=probabilities[label],
            )
            for index, label in enumerate(labels)
        ]


class FailingStanceClassifier:
    def predict(self, claim: str, evidence: list[dict]) -> list[StancePrediction]:
        raise StanceInferenceError("NLI model unavailable")


def _claims() -> list[Claim]:
    return [Claim("a", "Claim A"), Claim("b", "Claim B"), Claim("c", "Claim C")]


def _ratings() -> list[UserRating]:
    return [
        UserRating("a", "True"),
        UserRating("b", "True"),
        UserRating("c", "True"),
    ]


def _o3_ratings() -> list[UserRating]:
    return [
        UserRating("a", "True"),
        UserRating("a", "False"),
        UserRating("b", "True"),
        UserRating("b", "Mostly True"),
        UserRating("c", "True"),
        UserRating("c", "True"),
    ]


def _evidence(claim: str) -> list[dict]:
    return [
        {
            "title": f"{claim} evidence {index}",
            "url": f"https://example.test/{claim}/{index}",
            "content": f"Evidence {index} for {claim}",
            "relevance_score": 0.9 - (index * 0.1),
        }
        for index in range(4)
    ]


def _retrieve(claim: str, **kwargs: object) -> list[dict]:
    return _evidence(claim)


@pytest.mark.parametrize(
    ("objective", "pipeline_objective", "store"),
    [
        (
            ClaimRankObjective.REMOVE_FALSE_CLAIMS,
            Objective.O1,
            FakeStore(FakeModel("True", ("False", "True"))),
        ),
        (
            ClaimRankObjective.CORRECT_USER_MISCONCEPTIONS,
            Objective.O2,
            FakeStore(o2_model=FakeModel("Low", ("Low", "High"))),
        ),
        (
            ClaimRankObjective.REDUCE_USER_DISAGREEMENT,
            Objective.O3,
            FakeStore(),
        ),
    ],
)
def test_all_friendly_objectives_map_to_existing_pipeline(
    objective: ClaimRankObjective,
    pipeline_objective: Objective,
    store: FakeStore,
) -> None:
    service = ClaimRankService(
        artifact_store=store,
        retrieve=Mock(return_value=[]),
        stance_classifier=FakeStanceClassifier(),
    )

    run = service.rank_claims(_claims(), _ratings(), objective)

    assert run.objective is objective
    assert objective.pipeline_objective is pipeline_objective
    assert all(entry.objective is objective for entry in run.results)
    assert [entry.claim_id for entry in run.results] == ["a", "b", "c"]
    assert all(entry.conflict is not None for entry in run.results)
    assert all(
        entry.conflict.status is ECSStatus.NO_EVIDENCE
        for entry in run.results
    )
    assert all(entry.ecs_score is None for entry in run.results)
    assert run.fusion_method == "reciprocal_rank_fusion"
    assert run.rrf_k == 60.0


@pytest.mark.parametrize("objective", ["O3", Objective.O3])
def test_existing_objective_codes_are_accepted(objective: str | Objective) -> None:
    service = ClaimRankService(
        retrieve=Mock(return_value=[]),
        stance_classifier=FakeStanceClassifier(),
    )

    run = service.rank_claims(_claims(), _ratings(), objective)

    assert run.objective is ClaimRankObjective.REDUCE_USER_DISAGREEMENT


def test_invalid_objective_is_reported() -> None:
    service = ClaimRankService(
        retrieve=Mock(return_value=[]),
        stance_classifier=FakeStanceClassifier(),
    )

    with pytest.raises(ClaimRankServiceError) as exc_info:
        service.rank_claims(_claims(), _ratings(), "remove_fake_claims")

    assert exc_info.value.stage == "objective"


def test_original_ranks_and_scores_are_preserved_alongside_final_priority() -> None:
    service = ClaimRankService(
        retrieve=_retrieve,
        stance_classifier=FakeStanceClassifier(),
    )

    run = service.rank_claims(
        _claims(),
        _o3_ratings(),
        ClaimRankObjective.REDUCE_USER_DISAGREEMENT,
    )

    # O3 baseline: A (2.0), B (0.125), C (0.0).
    # ECS: B (1.0), C (0.5), A (0.0); RRF therefore promotes B above A.
    assert [entry.claim_id for entry in run.results] == ["b", "a", "c"]
    by_id = {entry.claim_id: entry for entry in run.results}

    assert by_id["a"].original.rank == 1
    assert by_id["a"].original.disputability == pytest.approx(2.0)
    assert by_id["a"].final.rank == 2
    assert by_id["a"].final.score == pytest.approx(1 / 61 + 1 / 63)

    assert by_id["b"].original.rank == 2
    assert by_id["b"].original.disputability == pytest.approx(0.125)
    assert by_id["b"].final.rank == 1
    assert by_id["b"].final.score == pytest.approx(1 / 62 + 1 / 61)

    assert by_id["a"].ecs_score == pytest.approx(0.0)
    assert by_id["b"].ecs_score == pytest.approx(1.0)
    assert by_id["c"].ecs_score == pytest.approx(0.5)
    assert by_id["a"].ecs_rank == 3
    assert by_id["b"].ecs_rank == 1
    assert by_id["c"].ecs_rank == 2

    assert by_id["a"].conflict is not None
    assert by_id["a"].conflict.status is ECSStatus.COMPUTED
    assert by_id["a"].conflict.counts.supports == 2
    assert by_id["a"].conflict.counts.refutes == 0
    assert len(by_id["a"].evidence) == 4
    assert by_id["a"].evidence[0]["url"] == "https://example.test/Claim A/0"


def test_retrieval_failure_is_exposed_without_fabricated_evidence() -> None:
    def failing_retrieve(claim: str, **kwargs: object) -> list[dict]:
        raise EvidenceSearchError("search unavailable")

    service = ClaimRankService(
        retrieve=failing_retrieve,
        stance_classifier=FakeStanceClassifier(),
    )

    run = service.rank_claims(
        _claims(),
        _ratings(),
        ClaimRankObjective.REDUCE_USER_DISAGREEMENT,
    )

    assert [entry.claim_id for entry in run.results] == ["a", "b", "c"]
    for entry in run.results:
        assert entry.evidence == ()
        assert entry.conflict is None
        assert entry.ecs_score is None
        assert entry.retrieval_error is not None
        assert entry.retrieval_error.error_type == "EvidenceSearchError"


def test_nli_failure_is_exposed_without_fabricated_ecs() -> None:
    service = ClaimRankService(
        retrieve=_retrieve,
        stance_classifier=FailingStanceClassifier(),
    )

    run = service.rank_claims(
        _claims(),
        _ratings(),
        ClaimRankObjective.REDUCE_USER_DISAGREEMENT,
    )

    assert [entry.claim_id for entry in run.results] == ["a", "b", "c"]
    for entry in run.results:
        assert entry.conflict is not None
        assert entry.conflict.status is ECSStatus.NLI_ERROR
        assert entry.ecs_score is None
        assert entry.ecs_rank is None
        assert entry.ecs_error is not None
        assert entry.ecs_error.error_type == "StanceInferenceError"
        assert entry.final.score == pytest.approx(1 / (60 + entry.original.rank))


def test_invalid_claim_input_is_reported() -> None:
    service = ClaimRankService(
        retrieve=Mock(return_value=[]),
        stance_classifier=FakeStanceClassifier(),
    )

    with pytest.raises(ClaimRankServiceError) as exc_info:
        service.rank_claims(
            [Claim("a", "Claim A"), Claim("a", "Duplicate claim")],
            [UserRating("a", "True")],
            ClaimRankObjective.REDUCE_USER_DISAGREEMENT,
        )

    assert exc_info.value.stage == "input"


def test_missing_model_raises_a_clear_service_error(tmp_path) -> None:
    paths = ArtifactPaths(
        o1_model=tmp_path / "missing_o1.pkl",
        o2_model=tmp_path / "missing_o2.pkl",
        scaler=tmp_path / "missing_scaler.pkl",
    )
    retrieve = Mock(return_value=[])
    service = ClaimRankService(
        artifact_store=ArtifactStore(paths),
        retrieve=retrieve,
        stance_classifier=FakeStanceClassifier(),
    )

    with pytest.raises(ClaimRankServiceError) as exc_info:
        service.rank_claims(
            _claims(),
            _ratings(),
            ClaimRankObjective.REMOVE_FALSE_CLAIMS,
        )

    assert exc_info.value.stage == "prioritization_models"
    assert "not found" in exc_info.value.message.lower()
    retrieve.assert_not_called()
