"""Focused tests for in-memory prioritization run management."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import prioritization_runs as runs_api
from api.main import app
from src.pipeline import (
    ClaimRankEntry,
    ClaimRankObjective,
    ClaimRankRun,
    ClaimRankServiceError,
    FinalPriority,
    OriginalPriority,
)

client = TestClient(app)

RUNS_URL = "/api/v1/prioritization-runs"
CLAIMS_CSV = b"claim_id,claim\n1,Claim One\n2,Claim Two\n"
PERCEPTION_CSV = b"claim_id,rating\n1,True\n2,False\n"


class FakeService:
    def __init__(
        self,
        *,
        result: ClaimRankRun | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[list, list, ClaimRankObjective]] = []

    def rank_claims(self, claims, ratings, objective):
        self.calls.append((list(claims), list(ratings), objective))
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _fake_run() -> ClaimRankRun:
    objective = ClaimRankObjective.REDUCE_USER_DISAGREEMENT
    entries = tuple(
        ClaimRankEntry(
            claim_id=claim_id,
            claim=f"Claim {claim_id}",
            objective=objective,
            original=OriginalPriority(rank=rank),
            final=FinalPriority(rank=rank, score=1.0 / (60 + rank)),
            ecs_score=None,
            ecs_rank=None,
            evidence=(),
            conflict=None,
            retrieval_error=None,
            ecs_error=None,
        )
        for rank, claim_id in enumerate(("1", "2"), start=1)
    )
    return ClaimRankRun(
        objective=objective,
        results=entries,
        fusion_method="reciprocal_rank_fusion",
        rrf_k=60.0,
    )


def _override_service(fake: FakeService) -> None:
    app.dependency_overrides[runs_api.get_claimrank_service] = lambda: fake


def _create_run(
    claims: bytes = CLAIMS_CSV,
    perception: bytes = PERCEPTION_CSV,
    objective: str = "reduce_user_disagreement",
):
    return client.post(
        RUNS_URL,
        files={
            "claims_file": ("claims.csv", claims, "text/csv"),
            "perception_file": ("perception.csv", perception, "text/csv"),
        },
        data={"objective": objective},
    )


@pytest.fixture(autouse=True)
def clean_run_state():
    runs_api.run_store.clear()
    runs_api.get_claimrank_service.cache_clear()
    yield
    app.dependency_overrides.pop(runs_api.get_claimrank_service, None)
    runs_api.run_store.clear()
    runs_api.get_claimrank_service.cache_clear()


def test_successful_run_queues_then_completes() -> None:
    fake = FakeService(result=_fake_run())
    _override_service(fake)

    response = _create_run()

    assert response.status_code == 202
    initial = response.json()
    assert initial["status"] == "queued"
    assert initial["stage"] == "queued"
    assert initial["claim_count"] == 2
    assert initial["rating_count"] == 2
    run_id = initial["run_id"]
    assert run_id.startswith("run_")

    status_response = client.get(f"{RUNS_URL}/{run_id}/status")
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["status"] == "completed"
    assert status["stage"] == "completed"
    assert status["progress"] == 1.0
    assert status["result_count"] == 2
    assert status["error"] is None
    assert status["started_at"] is not None
    assert status["completed_at"] is not None

    assert len(fake.calls) == 1
    claims, ratings, objective = fake.calls[0]
    assert [claim.claim_id for claim in claims] == ["1", "2"]
    assert [rating.rating for rating in ratings] == ["True", "False"]
    assert objective is ClaimRankObjective.REDUCE_USER_DISAGREEMENT


def test_status_transitions_queued_processing_completed(monkeypatch) -> None:
    monkeypatch.setattr(runs_api, "_execute_run", lambda *args, **kwargs: None)
    fake = FakeService(result=_fake_run())
    _override_service(fake)

    response = _create_run()
    run_id = response.json()["run_id"]

    queued = client.get(f"{RUNS_URL}/{run_id}/status").json()
    assert queued["status"] == "queued"
    assert queued["progress"] == 0.0

    runs_api.run_store.mark_processing(run_id)
    processing = client.get(f"{RUNS_URL}/{run_id}/status").json()
    assert processing["status"] == "processing"
    assert processing["stage"] == "running_pipeline"
    assert processing["progress"] == 0.1
    assert processing["started_at"] is not None

    runs_api.run_store.mark_completed(run_id, _fake_run())
    completed = client.get(f"{RUNS_URL}/{run_id}/status").json()
    assert completed["status"] == "completed"
    assert completed["progress"] == 1.0
    assert completed["result_count"] == 2


def test_invalid_objective_is_rejected_before_run_creation() -> None:
    fake = FakeService(result=_fake_run())
    _override_service(fake)

    response = _create_run(objective="remove_fake_claims")

    assert response.status_code == 422
    assert response.json()["detail"]["message"] == "Invalid objective."
    assert fake.calls == []


def test_missing_required_columns_are_rejected() -> None:
    fake = FakeService(result=_fake_run())
    _override_service(fake)

    response = _create_run(claims=b"claim_id\n1\n")

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["message"] == "Claims CSV validation failed."
    assert any("claim" in error for error in detail["errors"])
    assert fake.calls == []


@pytest.mark.parametrize(
    ("claims", "detail_fragment"),
    [
        (b"", "CSV file is empty."),
        (b'claim_id,claim\n1,"unterminated\n', "Malformed CSV"),
    ],
)
def test_empty_or_malformed_csv_is_rejected(
    claims: bytes,
    detail_fragment: str,
) -> None:
    fake = FakeService(result=_fake_run())
    _override_service(fake)

    response = _create_run(claims=claims)

    assert response.status_code == 400
    assert detail_fragment in response.json()["detail"]
    assert fake.calls == []


def test_cross_file_claim_id_mismatch_is_rejected() -> None:
    fake = FakeService(result=_fake_run())
    _override_service(fake)

    response = _create_run(
        perception=b"claim_id,rating\n999,True\n",
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["message"] == "Claims and perception data are inconsistent."
    assert any("999" in error for error in detail["errors"])
    assert fake.calls == []


def test_unknown_run_id_returns_404() -> None:
    response = client.get(f"{RUNS_URL}/run_missing/status")

    assert response.status_code == 404
    assert response.json()["detail"] == "Prioritization run not found."


def test_service_failure_marks_run_failed_without_traceback() -> None:
    fake = FakeService(
        error=ClaimRankServiceError("prioritization_models", "model file missing")
    )
    _override_service(fake)

    response = _create_run()
    run_id = response.json()["run_id"]

    status_response = client.get(f"{RUNS_URL}/{run_id}/status")
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["status"] == "failed"
    assert status["stage"] == "failed"
    assert status["result_count"] is None
    assert status["completed_at"] is not None
    assert status["error"] == {
        "stage": "prioritization_models",
        "type": "ClaimRankServiceError",
        "message": "model file missing",
    }
    assert "Traceback" not in status_response.text


def test_unexpected_failure_is_sanitized() -> None:
    fake = FakeService(error=RuntimeError("secret internal detail"))
    _override_service(fake)

    response = _create_run()
    run_id = response.json()["run_id"]

    status_response = client.get(f"{RUNS_URL}/{run_id}/status")
    status = status_response.json()
    assert status["status"] == "failed"
    assert status["error"] == {
        "stage": "pipeline",
        "type": "RuntimeError",
        "message": runs_api.UNEXPECTED_ERROR_MESSAGE,
    }
    assert "secret internal detail" not in status_response.text
    assert "Traceback" not in status_response.text
