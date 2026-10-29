"""Focused tests for user-triggered FactLLaMA fact-check jobs."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import fact_checks as fact_checks_api
from api import prioritization_runs as runs_api
from api.main import app
from src.factllama import FactLLaMAPredictor
from src.pipeline import (
    ClaimRankEntry,
    ClaimRankObjective,
    ClaimRankRun,
    FactLLaMAErrorInfo,
    FactLLaMASelectionService,
    FactLLaMAVerificationResult,
    FinalPriority,
    OriginalPriority,
    RetrievalErrorInfo,
)

client = TestClient(app)

RUNS_URL = "/api/v1/prioritization-runs"
JOBS_URL = "/api/v1/fact-check-jobs"

EVIDENCE = (
    {
        "title": "Source one",
        "url": "https://example.test/one",
        "content": "Alpha claim is supported by the first source.",
    },
    {
        "title": "Source two",
        "url": "https://example.test/two",
        "content": "A second source also discusses the alpha claim.",
    },
)


class FakeFactLLaMAService:
    def __init__(
        self,
        *,
        verification: FactLLaMAVerificationResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.verification = verification
        self.error = error
        self.calls: list[tuple[list, object]] = []

    def verify_selected_claim(self, results, claim_id):
        values = list(results)
        self.calls.append((values, claim_id))
        if self.error is not None:
            raise self.error
        assert self.verification is not None
        return self.verification


def _verification(
    *,
    claim_id: str | int = "1",
    claim: str = "Alpha claim",
    verdict: str | None = "true",
    raw_output: str | None = "true",
    error: FactLLaMAErrorInfo | None = None,
    evidence=EVIDENCE,
) -> FactLLaMAVerificationResult:
    return FactLLaMAVerificationResult(
        claim_id=claim_id,
        claim=claim,
        evidence=evidence,
        verdict=verdict,
        raw_output=raw_output,
        error=error,
    )


def _entry(
    claim_id: str | int = "1",
    *,
    claim: str = "Alpha claim",
    evidence=EVIDENCE,
    retrieval_error: RetrievalErrorInfo | None = None,
) -> ClaimRankEntry:
    return ClaimRankEntry(
        claim_id=claim_id,
        claim=claim,
        objective=ClaimRankObjective.REDUCE_USER_DISAGREEMENT,
        original=OriginalPriority(rank=1, disputability=0.4),
        final=FinalPriority(rank=1, score=1 / 61),
        ecs_score=None,
        ecs_rank=None,
        evidence=evidence,
        conflict=None,
        retrieval_error=retrieval_error,
        ecs_error=None,
    )


def _create_run(
    *entries: ClaimRankEntry,
    status: str = "completed",
) -> str:
    selected = entries or (_entry(),)
    run = runs_api.run_store.create(
        objective=ClaimRankObjective.REDUCE_USER_DISAGREEMENT,
        claim_count=len(selected),
        rating_count=len(selected),
    )
    if status == "completed":
        runs_api.run_store.mark_completed(
            run.run_id,
            ClaimRankRun(
                objective=ClaimRankObjective.REDUCE_USER_DISAGREEMENT,
                results=tuple(selected),
                fusion_method="reciprocal_rank_fusion",
                rrf_k=60.0,
            ),
        )
    elif status == "processing":
        runs_api.run_store.mark_processing(run.run_id)
    elif status == "failed":
        runs_api.run_store.mark_failed(
            run.run_id,
            stage="running_pipeline",
            error_type="ClaimRankServiceError",
            message="model file missing",
        )
    return run.run_id


def _override(service) -> None:
    app.dependency_overrides[fact_checks_api.get_factllama_service] = (
        lambda: service
    )


class RecordingGenerator:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages):
        self.calls.append(list(messages))
        return self.output


def _post(run_id: str, claim_id: str = "1"):
    return client.post(
        f"{RUNS_URL}/{run_id}/claims/{claim_id}/fact-check"
    )


def _get_job(job_id: str):
    return client.get(f"{JOBS_URL}/{job_id}")


@pytest.fixture(autouse=True)
def clean_state():
    runs_api.run_store.clear()
    fact_checks_api.fact_check_store.clear()
    fact_checks_api.reset_factllama_service()
    yield
    app.dependency_overrides.pop(fact_checks_api.get_factllama_service, None)
    runs_api.run_store.clear()
    fact_checks_api.fact_check_store.clear()
    fact_checks_api.reset_factllama_service()


def test_fact_check_job_completes_with_verdict_and_stored_evidence() -> None:
    run_id = _create_run()
    service = FakeFactLLaMAService(verification=_verification())
    _override(service)

    response = _post(run_id)

    assert response.status_code == 202
    initial = response.json()
    assert initial["job_id"].startswith("job_")
    assert initial["run_id"] == run_id
    assert initial["claim_id"] == "1"
    assert initial["claim"] == "Alpha claim"
    assert initial["status"] == "queued"
    assert initial["verdict"] is None
    assert initial["explanation"] is None
    assert initial["evidence_references"] == []
    assert len(initial["limitations"]) >= 3
    assert initial["error"] is None

    response = _get_job(initial["job_id"])

    assert response.status_code == 200
    job = response.json()
    assert job["status"] == "completed"
    assert job["verdict"] == "true"
    assert job["raw_output"] == "true"
    assert job["explanation"] is None
    assert job["error"] is None
    assert job["started_at"] is not None
    assert job["completed_at"] is not None

    references = job["evidence_references"]
    assert [reference["index"] for reference in references] == [1, 2]
    assert references[0]["title"] == "Source one"
    assert references[0]["url"] == "https://example.test/one"
    assert (
        references[0]["content"]
        == "Alpha claim is supported by the first source."
    )

    # FactLLaMA received exactly the stored evidence through the existing
    # selection-service contract; no retrieval was performed.
    assert len(service.calls) == 1
    results, selected_id = service.calls[0]
    assert selected_id == "1"
    assert len(results) == 1
    assert results[0].claim_id == "1"
    assert results[0].objective.value == "O3"
    assert results[0].evidence == list(EVIDENCE)
    assert results[0].retrieval_error is None


def test_real_selection_service_uses_only_stored_evidence() -> None:
    run_id = _create_run()
    generator = RecordingGenerator("true")
    service = FactLLaMASelectionService(
        predictor=FactLLaMAPredictor(generator=generator)
    )
    _override(service)

    response = _post(run_id)

    assert response.status_code == 202
    job = _get_job(response.json()["job_id"]).json()
    assert job["status"] == "completed"
    assert job["verdict"] == "true"
    assert len(generator.calls) == 1

    prompt = generator.calls[0][1]["content"]
    assert "Alpha claim" in prompt
    assert "Alpha claim is supported by the first source." in prompt
    assert "A second source also discusses the alpha claim." in prompt
    # URLs and titles are stored as references but are not sent to the model.
    assert "https://example.test/one" not in prompt
    assert "Source one" not in prompt


def test_fact_check_matches_stable_numeric_claim_id() -> None:
    run_id = _create_run(_entry(7))
    service = FakeFactLLaMAService(
        verification=_verification(claim_id=7)
    )
    _override(service)

    response = _post(run_id, "7")

    assert response.status_code == 202
    job = _get_job(response.json()["job_id"]).json()
    assert job["claim_id"] == 7
    assert job["status"] == "completed"
    assert service.calls[0][1] == 7


def test_unparseable_output_marks_job_failed() -> None:
    run_id = _create_run()
    service = FakeFactLLaMAService(
        verification=_verification(
            verdict=None,
            raw_output="unsure",
            error=FactLLaMAErrorInfo(
                "VerdictParseError",
                "FactLLaMA output did not contain one unambiguous supported "
                "verdict.",
            ),
        )
    )
    _override(service)

    response = _post(run_id)

    assert response.status_code == 202
    job = _get_job(response.json()["job_id"]).json()
    assert job["status"] == "failed"
    assert job["verdict"] is None
    assert job["raw_output"] == "unsure"
    assert job["error"]["type"] == "VerdictParseError"
    assert "unambiguous" in job["error"]["message"]
    assert len(job["evidence_references"]) == 2
    assert "Traceback" not in str(job)


def test_inference_failure_marks_job_failed_without_traceback() -> None:
    run_id = _create_run()
    service = FakeFactLLaMAService(
        verification=_verification(
            verdict=None,
            raw_output=None,
            error=FactLLaMAErrorInfo(
                "ModelLoadError",
                "adapter weights are unavailable",
            ),
        )
    )
    _override(service)

    response = _post(run_id)

    assert response.status_code == 202
    job = _get_job(response.json()["job_id"]).json()
    assert job["status"] == "failed"
    assert job["error"] == {
        "type": "ModelLoadError",
        "message": "adapter weights are unavailable",
    }
    assert "Traceback" not in str(job)


def test_unexpected_failure_is_sanitized() -> None:
    run_id = _create_run()
    service = FakeFactLLaMAService(
        error=RuntimeError("secret path C:\\private\\weights")
    )
    _override(service)

    response = _post(run_id)

    assert response.status_code == 202
    job = _get_job(response.json()["job_id"]).json()
    assert job["status"] == "failed"
    assert job["error"] == {
        "type": "UnexpectedError",
        "message": "Fact check failed unexpectedly.",
    }
    assert "secret path" not in str(job)
    assert "Traceback" not in str(job)


def test_unknown_run_and_job_return_404() -> None:
    unknown_run = _post("run_missing")
    assert unknown_run.status_code == 404
    assert unknown_run.json()["detail"] == "Prioritization run not found."

    unknown_job = _get_job("job_missing")
    assert unknown_job.status_code == 404
    assert unknown_job.json()["detail"] == "Fact-check job not found."


@pytest.mark.parametrize(
    ("status", "message"),
    [
        ("processing", "Prioritization run is not complete."),
        ("failed", "Prioritization run failed."),
    ],
)
def test_incomplete_or_failed_run_cannot_start_fact_check(
    status: str,
    message: str,
) -> None:
    run_id = _create_run(status=status)

    response = _post(run_id)

    assert response.status_code == 409
    assert response.json()["detail"]["message"] == message
    assert fact_checks_api.fact_check_store.snapshot("job_missing") is None


def test_missing_claim_returns_404() -> None:
    run_id = _create_run()

    response = _post(run_id, "999")

    assert response.status_code == 404
    assert (
        response.json()["detail"]
        == "Claim not found in this prioritization run."
    )


def test_claim_with_retrieval_error_returns_422() -> None:
    run_id = _create_run(
        _entry(
            retrieval_error=RetrievalErrorInfo(
                "EvidenceRetrievalError",
                "search service unavailable",
            )
        )
    )

    response = _post(run_id)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "evidence retrieval failed" in detail["message"]
    assert detail["retrieval_error"] == {
        "error_type": "EvidenceRetrievalError",
        "message": "search service unavailable",
    }


def test_claim_without_evidence_returns_422() -> None:
    run_id = _create_run(_entry(evidence=()))

    response = _post(run_id)

    assert response.status_code == 422
    assert (
        "no usable stored evidence"
        in response.json()["detail"]["message"]
    )


def test_claim_with_unusable_evidence_returns_422() -> None:
    run_id = _create_run(
        _entry(evidence=({"title": "Metadata only"},))
    )

    response = _post(run_id)

    assert response.status_code == 422
    assert (
        "no usable stored evidence"
        in response.json()["detail"]["message"]
    )


def test_empty_claim_returns_422() -> None:
    run_id = _create_run(_entry(claim="   "))

    response = _post(run_id)

    assert response.status_code == 422
    assert "empty claim" in response.json()["detail"]["message"]


def test_factllama_service_is_cached_per_process() -> None:
    assert (
        fact_checks_api.get_factllama_service()
        is fact_checks_api.get_factllama_service()
    )
