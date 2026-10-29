"""Focused tests for completed-run results, summary, and claim APIs."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from api import prioritization_runs as runs_api
from api.main import app
from src.evidence_conflict import (
    ECSStatus,
    EvidenceConflictResult,
    StanceCounts,
    StanceLabel,
    StancePrediction,
    StanceProbabilities,
)
from src.evidence_conflict.scoring import compute_evidence_conflict
from src.pipeline import (
    ClaimRankEntry,
    ClaimRankObjective,
    ClaimRankRun,
    ClaimRankServiceError,
    ECSErrorInfo,
    FinalPriority,
    OriginalPriority,
    RetrievalErrorInfo,
    parse_objective,
)

client = TestClient(app)

RUNS_URL = "/api/v1/prioritization-runs"
CLAIMS_CSV = (
    b"claim_id,claim\n"
    b"1,Alpha claim\n"
    b"2,Beta claim\n"
    b"3,Gamma claim\n"
    b"4,Delta claim\n"
)
PERCEPTION_CSV = (
    b"claim_id,rating\n"
    b"1,True\n"
    b"2,True\n"
    b"3,True\n"
    b"4,True\n"
)

_PROBABILITIES = {
    StanceLabel.SUPPORTS: StanceProbabilities(0.9, 0.05, 0.05),
    StanceLabel.REFUTES: StanceProbabilities(0.05, 0.9, 0.05),
    StanceLabel.NOT_ENOUGH_INFO: StanceProbabilities(0.1, 0.1, 0.8),
}


class FakeService:
    def __init__(self, *, result: ClaimRankRun | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error

    def rank_claims(self, claims, ratings, objective):
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _prediction(index: int, label: StanceLabel) -> StancePrediction:
    return StancePrediction(
        evidence_index=index,
        label=label,
        confidence=0.9,
        probabilities=_PROBABILITIES[label],
    )


def _computed_conflict(claim: str, labels: tuple[StanceLabel, ...]) -> EvidenceConflictResult:
    return compute_evidence_conflict(
        [_prediction(index, label) for index, label in enumerate(labels)],
        claim_text=claim,
    )


def _evidence(claim: str, count: int) -> tuple[dict, ...]:
    return tuple(
        {
            "title": f"{claim} source {index}",
            "url": f"https://example.test/{claim}/{index}",
            "content": f"Evidence {index} for {claim}",
            "relevance_score": 0.9 - index * 0.1,
        }
        for index in range(count)
    )


def _entry(
    claim_id: str,
    claim: str,
    objective: ClaimRankObjective,
    *,
    initial_rank: int,
    final_rank: int,
    conflict: EvidenceConflictResult | None,
    evidence_count: int,
    ecs_rank: int | None,
    predicted_truth: str | None = None,
    predicted_tpb: str | None = None,
    disputability: float | None = None,
    retrieval_error: RetrievalErrorInfo | None = None,
    ecs_error: ECSErrorInfo | None = None,
) -> ClaimRankEntry:
    return ClaimRankEntry(
        claim_id=claim_id,
        claim=claim,
        objective=objective,
        original=OriginalPriority(
            rank=initial_rank,
            predicted_truth=predicted_truth,
            predicted_tpb=predicted_tpb,
            disputability=disputability,
        ),
        final=FinalPriority(
            rank=final_rank,
            score=1.0 / (60.0 + final_rank),
        ),
        ecs_score=conflict.ecs if conflict is not None else None,
        ecs_rank=ecs_rank,
        evidence=_evidence(claim, evidence_count),
        conflict=conflict,
        retrieval_error=retrieval_error,
        ecs_error=ecs_error,
    )


def _rich_run(objective: ClaimRankObjective) -> ClaimRankRun:
    truth = {"1": "True", "2": "False", "3": "True", "4": "False"}
    tpb = {"1": "High", "2": "Low", "3": "High", "4": "Low"}
    disputability = {"1": 0.5, "2": 2.0, "3": 0.0, "4": 0.3}

    conflict_one = _computed_conflict(
        "Alpha claim",
        (
            StanceLabel.SUPPORTS,
            StanceLabel.SUPPORTS,
            StanceLabel.REFUTES,
            StanceLabel.REFUTES,
            StanceLabel.NOT_ENOUGH_INFO,
        ),
    )
    conflict_two = _computed_conflict(
        "Beta claim",
        (StanceLabel.SUPPORTS, StanceLabel.SUPPORTS),
    )
    conflict_three = _computed_conflict("Gamma claim", ())
    conflict_four = EvidenceConflictResult(
        claim="Delta claim",
        predictions=(),
        counts=StanceCounts(),
        status=ECSStatus.NLI_ERROR,
    )

    def values(claim_id: str) -> dict[str, object]:
        return {
            "predicted_truth": (
                truth[claim_id]
                if objective is ClaimRankObjective.REMOVE_FALSE_CLAIMS
                else None
            ),
            "predicted_tpb": (
                tpb[claim_id]
                if objective is ClaimRankObjective.CORRECT_USER_MISCONCEPTIONS
                else None
            ),
            "disputability": (
                disputability[claim_id]
                if objective is ClaimRankObjective.REDUCE_USER_DISAGREEMENT
                else None
            ),
        }

    entries = (
        _entry(
            "1",
            "Alpha claim",
            objective,
            initial_rank=2,
            final_rank=1,
            conflict=conflict_one,
            evidence_count=5,
            ecs_rank=1,
            **values("1"),
        ),
        _entry(
            "2",
            "Beta claim",
            objective,
            initial_rank=1,
            final_rank=2,
            conflict=conflict_two,
            evidence_count=2,
            ecs_rank=2,
            **values("2"),
        ),
        _entry(
            "3",
            "Gamma claim",
            objective,
            initial_rank=4,
            final_rank=3,
            conflict=conflict_three,
            evidence_count=0,
            ecs_rank=None,
            **values("3"),
        ),
        _entry(
            "4",
            "Delta claim",
            objective,
            initial_rank=3,
            final_rank=4,
            conflict=conflict_four,
            evidence_count=2,
            ecs_rank=None,
            ecs_error=ECSErrorInfo(
                "StanceInferenceError",
                "NLI model unavailable",
            ),
            **values("4"),
        ),
    )
    return ClaimRankRun(
        objective=objective,
        results=entries,
        fusion_method="reciprocal_rank_fusion",
        rrf_k=60.0,
    )


def _override_service(fake: FakeService) -> None:
    app.dependency_overrides[runs_api.get_claimrank_service] = lambda: fake


def _post_run_result(result: ClaimRankRun, objective: str) -> str:
    _override_service(FakeService(result=result))
    response = client.post(
        RUNS_URL,
        files={
            "claims_file": ("claims.csv", CLAIMS_CSV, "text/csv"),
            "perception_file": ("perception.csv", PERCEPTION_CSV, "text/csv"),
        },
        data={"objective": objective},
    )
    assert response.status_code == 202
    return response.json()["run_id"]


def _create_run(objective: str = "reduce_user_disagreement") -> str:
    return _post_run_result(
        _rich_run(parse_objective(objective)),
        objective,
    )


@pytest.fixture(autouse=True)
def clean_run_state():
    runs_api.run_store.clear()
    runs_api.get_claimrank_service.cache_clear()
    yield
    app.dependency_overrides.pop(runs_api.get_claimrank_service, None)
    runs_api.run_store.clear()
    runs_api.get_claimrank_service.cache_clear()


def test_results_returns_complete_serialized_entries() -> None:
    run_id = _create_run()

    response = client.get(f"{RUNS_URL}/{run_id}/results")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == run_id
    assert payload["status"] == "completed"
    assert payload["objective"] == "reduce_user_disagreement"
    assert payload["fusion_method"] == "reciprocal_rank_fusion"
    assert payload["rrf_k"] == 60.0
    assert payload["result_count"] == 4

    first = payload["results"][0]
    assert first["claim_id"] == "1"
    assert first["claim"] == "Alpha claim"
    assert first["original"]["rank"] == 2
    assert first["original"]["disputability"] == 0.5
    assert first["final"]["rank"] == 1
    assert first["final"]["score"] == pytest.approx(1 / 61)
    assert first["ecs_score"] == pytest.approx(0.8)
    assert first["ecs_rank"] == 1
    assert first["conflict"]["status"] == "computed"
    assert first["conflict"]["counts"] == {
        "supports": 2,
        "refutes": 2,
        "not_enough_info": 1,
    }
    assert first["conflict"]["predictions"][0]["label"] == "supports"
    assert first["conflict"]["predictions"][0]["probabilities"] == {
        "supports": 0.9,
        "refutes": 0.05,
        "not_enough_info": 0.05,
    }
    assert first["evidence"][0]["url"] == "https://example.test/Alpha claim/0"
    assert first["retrieval_error"] is None
    assert first["ecs_error"] is None

    last = payload["results"][3]
    assert last["ecs_score"] is None
    assert last["conflict"]["status"] == "nli_error"
    assert last["ecs_error"] == {
        "error_type": "StanceInferenceError",
        "message": "NLI model unavailable",
    }
    assert "Traceback" not in response.text


def test_summary_uses_only_stored_result_values() -> None:
    run_id = _create_run()

    response = client.get(f"{RUNS_URL}/{run_id}/summary")

    assert response.status_code == 200
    summary = response.json()
    assert summary["status"] == "completed"
    assert summary["total_claims"] == 4
    assert summary["input_claim_count"] == 4
    assert summary["high_conflict_claim_count"] == 1
    assert summary["high_conflict_threshold"] == 0.7
    assert summary["conflicting_claim_count"] == 1
    assert summary["claims_with_evidence"] == 3
    assert summary["claims_without_evidence"] == 1
    assert summary["evidence_coverage"] == pytest.approx(0.75)
    assert summary["mean_stance_coverage"] == pytest.approx(0.9)
    assert summary["conflict_status_counts"]["computed"] == 2
    assert summary["conflict_status_counts"]["no_evidence"] == 1
    assert summary["conflict_status_counts"]["nli_error"] == 1
    assert summary["conflict_status_counts"]["retrieval_error"] == 0
    assert summary["component_error_counts"] == {"retrieval": 0, "ecs": 1}
    assert summary["priority_label_counts"] == {
        "predicted_truth": None,
        "predicted_tpb": None,
    }

    scores = summary["score_summaries"]
    assert scores["final_score"]["count"] == 4
    assert scores["final_score"]["min"] == pytest.approx(1 / 64)
    assert scores["final_score"]["max"] == pytest.approx(1 / 61)
    assert scores["ecs_score"]["count"] == 2
    assert scores["ecs_score"]["min"] == 0.0
    assert scores["ecs_score"]["max"] == pytest.approx(0.8)
    assert scores["ecs_score"]["mean"] == pytest.approx(0.4)
    assert scores["disputability"]["count"] == 4
    assert scores["disputability"]["mean"] == pytest.approx(0.7)


def test_export_returns_downloadable_csv_with_available_fields() -> None:
    run_id = _create_run()

    response = client.get(f"{RUNS_URL}/{run_id}/export")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    disposition = response.headers["content-disposition"]
    assert disposition.startswith("attachment;")
    assert f"claimrank_{run_id}.csv" in disposition

    reader = csv.DictReader(io.StringIO(response.text))
    assert reader.fieldnames == list(runs_api.EXPORT_HEADERS)
    rows = list(reader)
    assert len(rows) == 4

    first = rows[0]
    assert first["Claim ID"] == "1"
    assert first["Claim"] == "Alpha claim"
    assert first["Objective"] == "reduce_user_disagreement"
    assert first["Original Rank"] == "2"
    assert first["Original Predicted Truth"] == ""
    assert first["Original Predicted TPB"] == ""
    assert first["Original Disputability"] == "0.5"
    assert first["Final Rank"] == "1"
    assert first["Rank Change"] == "1"
    assert float(first["Final Priority Score (RRF)"]) == pytest.approx(1 / 61)
    assert float(first["ECS Score"]) == pytest.approx(0.8)
    assert first["ECS Rank"] == "1"
    assert first["Conflict Status"] == "computed"
    assert first["Supporting Evidence Count"] == "2"
    assert first["Refuting Evidence Count"] == "2"
    assert first["Not Enough Info Count"] == "1"
    assert float(first["Stance Coverage"]) == pytest.approx(0.8)
    assert float(first["Refutation Share"]) == pytest.approx(0.5)
    assert float(first["Support Share"]) == pytest.approx(0.5)
    assert float(first["Mean Stance Confidence"]) == pytest.approx(0.9)
    assert first["Evidence Count"] == "5"
    assert "https://example.test/Alpha claim/0" in first["Evidence URLs"]
    assert len(json.loads(first["Evidence (JSON)"])) == 5
    predictions = json.loads(first["Conflict Predictions (JSON)"])
    assert len(predictions) == 5
    assert predictions[0]["label"] == "supports"

    no_evidence = rows[2]
    assert no_evidence["Conflict Status"] == "no_evidence"
    assert no_evidence["ECS Score"] == ""
    assert no_evidence["Stance Coverage"] == ""
    assert no_evidence["Evidence Count"] == "0"
    assert no_evidence["Evidence URLs"] == ""
    assert no_evidence["Evidence (JSON)"] == "[]"

    failed_ecs = rows[3]
    assert failed_ecs["Conflict Status"] == "nli_error"
    assert failed_ecs["ECS Score"] == ""
    assert failed_ecs["ECS Rank"] == ""
    assert failed_ecs["Conflict Predictions (JSON)"] == "[]"
    assert failed_ecs["ECS Error"] == "StanceInferenceError: NLI model unavailable"
    assert failed_ecs["Retrieval Error"] == ""


def test_export_handles_csv_escaping_for_text_and_nested_data() -> None:
    run = _rich_run(ClaimRankObjective.REDUCE_USER_DISAGREEMENT)
    tricky_claim = 'He said, "hello"\nnext line'
    tricky_evidence = (
        {
            "title": 'Title, "quoted"',
            "url": "https://example.test/a,b",
            "content": 'Line one\nLine two, "quoted"',
            "relevance_score": 0.8,
        },
    )
    tricky_entry = replace(
        run.results[0],
        claim=tricky_claim,
        evidence=tricky_evidence,
    )
    run = replace(run, results=(tricky_entry,) + run.results[1:])
    run_id = _post_run_result(run, "reduce_user_disagreement")

    response = client.get(f"{RUNS_URL}/{run_id}/export")

    assert response.status_code == 200
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert len(rows) == 4
    assert rows[0]["Claim"] == tricky_claim
    assert rows[0]["Evidence URLs"] == "https://example.test/a,b"
    evidence = json.loads(rows[0]["Evidence (JSON)"])
    assert evidence[0]["title"] == 'Title, "quoted"'
    assert evidence[0]["content"] == 'Line one\nLine two, "quoted"'


def test_claims_support_search_pagination_and_sorting() -> None:
    run_id = _create_run()
    url = f"{RUNS_URL}/{run_id}/claims"

    search = client.get(f"{url}?q=beta").json()
    assert search["matched_claims"] == 1
    assert [claim["claim_id"] for claim in search["claims"]] == ["2"]

    page = client.get(f"{url}?offset=1&limit=2").json()
    assert page["total_claims"] == 4
    assert page["matched_claims"] == 4
    assert page["offset"] == 1
    assert page["limit"] == 2
    assert page["returned"] == 2
    assert page["has_more"] is True
    assert [claim["claim_id"] for claim in page["claims"]] == ["2", "3"]

    initial = client.get(f"{url}?ranking=initial").json()
    assert initial["sort"] == "initial_rank"
    assert initial["order"] == "asc"
    assert [claim["claim_id"] for claim in initial["claims"]] == ["2", "1", "4", "3"]

    ecs = client.get(f"{url}?sort=ecs_score").json()
    assert ecs["order"] == "desc"
    assert [claim["claim_id"] for claim in ecs["claims"]] == ["1", "2", "3", "4"]

    moved = client.get(f"{url}?sort=rank_change&order=desc").json()
    assert [claim["claim_id"] for claim in moved["claims"]] == ["1", "3", "2", "4"]


def test_claims_filter_by_conflict_and_evidence() -> None:
    run_id = _create_run()
    url = f"{RUNS_URL}/{run_id}/claims"

    high = client.get(f"{url}?conflict=high").json()
    assert [claim["claim_id"] for claim in high["claims"]] == ["1"]

    low = client.get(f"{url}?conflict=low").json()
    assert [claim["claim_id"] for claim in low["claims"]] == ["2"]

    no_evidence = client.get(f"{url}?conflict_status=no_evidence").json()
    assert [claim["claim_id"] for claim in no_evidence["claims"]] == ["3"]

    without_evidence = client.get(f"{url}?has_evidence=false").json()
    assert [claim["claim_id"] for claim in without_evidence["claims"]] == ["3"]

    with_bounds = client.get(f"{url}?min_ecs=0.5&max_ecs=1.0").json()
    assert [claim["claim_id"] for claim in with_bounds["claims"]] == ["1"]

    disputable = client.get(f"{url}?min_disputability=0.4").json()
    assert [claim["claim_id"] for claim in disputable["claims"]] == ["1", "2"]


def test_claims_filter_by_objective_specific_priority_label() -> None:
    run_id = _create_run("remove_false_claims")
    url = f"{RUNS_URL}/{run_id}/claims"

    false_claims = client.get(f"{url}?priority=false").json()
    assert [claim["claim_id"] for claim in false_claims["claims"]] == ["2", "4"]

    true_claims = client.get(f"{url}?priority=True").json()
    assert [claim["claim_id"] for claim in true_claims["claims"]] == ["1", "3"]

    high = client.get(f"{url}?priority=high")
    assert high.status_code == 422
    assert "True" in high.json()["detail"]["errors"][0]


def test_claims_filter_by_tpb_priority_label() -> None:
    run_id = _create_run("correct_user_misconceptions")
    url = f"{RUNS_URL}/{run_id}/claims"

    high = client.get(f"{url}?priority=High").json()
    assert [claim["claim_id"] for claim in high["claims"]] == ["1", "3"]

    low = client.get(f"{url}?priority=low").json()
    assert [claim["claim_id"] for claim in low["claims"]] == ["2", "4"]

    invalid = client.get(f"{url}?priority=True")
    assert invalid.status_code == 422
    assert "High" in invalid.json()["detail"]["errors"][0]


def test_priority_filter_is_rejected_for_disputability_objective() -> None:
    run_id = _create_run()

    response = client.get(f"{RUNS_URL}/{run_id}/claims?priority=high")

    assert response.status_code == 422
    assert "min_disputability" in response.json()["detail"]["errors"][0]


def test_claim_detail_uses_stable_id_and_includes_rank_change() -> None:
    run_id = _create_run()

    response = client.get(f"{RUNS_URL}/{run_id}/claims/4")

    assert response.status_code == 200
    payload = response.json()
    claim = payload["claim"]
    assert claim["claim_id"] == "4"
    assert claim["original"]["rank"] == 3
    assert claim["final"]["rank"] == 4
    assert claim["rank_change"] == -1
    assert len(claim["evidence"]) == 2
    assert claim["conflict"]["status"] == "nli_error"
    assert claim["ecs_score"] is None
    assert claim["ecs_error"]["error_type"] == "StanceInferenceError"


def test_unknown_run_and_claim_return_404() -> None:
    run_id = _create_run()

    for suffix in ("results", "summary", "claims", "claims/1", "export"):
        response = client.get(f"{RUNS_URL}/run_missing/{suffix}")
        assert response.status_code == 404
        assert response.json()["detail"] == "Prioritization run not found."

    missing_claim = client.get(f"{RUNS_URL}/{run_id}/claims/999")
    assert missing_claim.status_code == 404
    assert missing_claim.json()["detail"] == "Claim not found in this prioritization run."


def test_queued_and_processing_runs_do_not_return_results(monkeypatch) -> None:
    monkeypatch.setattr(runs_api, "_execute_run", lambda *args, **kwargs: None)
    _override_service(FakeService(result=_rich_run(ClaimRankObjective.REDUCE_USER_DISAGREEMENT)))

    response = client.post(
        RUNS_URL,
        files={
            "claims_file": ("claims.csv", CLAIMS_CSV, "text/csv"),
            "perception_file": ("perception.csv", PERCEPTION_CSV, "text/csv"),
        },
        data={"objective": "reduce_user_disagreement"},
    )
    run_id = response.json()["run_id"]

    for suffix in ("results", "summary", "claims", "claims/1", "export"):
        queued = client.get(f"{RUNS_URL}/{run_id}/{suffix}")
        assert queued.status_code == 409
        assert queued.json()["detail"]["message"] == "Prioritization run is not complete."
        assert queued.json()["detail"]["status"] == "queued"

    runs_api.run_store.mark_processing(run_id)
    processing = client.get(f"{RUNS_URL}/{run_id}/results")
    assert processing.status_code == 409
    assert processing.json()["detail"]["status"] == "processing"
    assert processing.json()["detail"]["stage"] == "running_pipeline"


def test_failed_run_returns_consistent_clear_error() -> None:
    _override_service(
        FakeService(
            error=ClaimRankServiceError(
                "prioritization_models",
                "model file missing",
            )
        )
    )
    response = client.post(
        RUNS_URL,
        files={
            "claims_file": ("claims.csv", CLAIMS_CSV, "text/csv"),
            "perception_file": ("perception.csv", PERCEPTION_CSV, "text/csv"),
        },
        data={"objective": "reduce_user_disagreement"},
    )
    run_id = response.json()["run_id"]

    for suffix in ("results", "summary", "claims", "claims/1", "export"):
        failed = client.get(f"{RUNS_URL}/{run_id}/{suffix}")
        assert failed.status_code == 409
        detail = failed.json()["detail"]
        assert detail["message"] == "Prioritization run failed."
        assert detail["status"] == "failed"
        assert detail["error"] == {
            "stage": "prioritization_models",
            "type": "ClaimRankServiceError",
            "message": "model file missing",
        }
        assert "Traceback" not in failed.text


@pytest.mark.parametrize(
    "query",
    [
        "limit=0",
        "limit=101",
        "offset=-1",
        "sort=unknown",
        "order=sideways",
        "ranking=middle",
        "conflict=extreme",
        "conflict_status=unknown",
        "min_ecs=1.5",
        "min_ecs=0.8&max_ecs=0.2",
        "min_disputability=-1",
        "min_disputability=2&max_disputability=1",
    ],
)
def test_invalid_filters_return_422(query: str) -> None:
    run_id = _create_run()

    response = client.get(f"{RUNS_URL}/{run_id}/claims?{query}")

    assert response.status_code == 422


def test_q_and_search_cannot_be_combined() -> None:
    run_id = _create_run()

    response = client.get(f"{RUNS_URL}/{run_id}/claims?q=alpha&search=beta")

    assert response.status_code == 422
    assert response.json()["detail"]["message"] == "Invalid search filter."
