"""Offline tests for claim-ranking dataset validation and baseline extraction.

All fixtures are synthetic and exist only in memory. They are not saved as
evaluation results and do not imply real-world performance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from evaluation.claim_ranking.dataset import (
    DatasetValidationError,
    baseline_ranking,
    dataset_summary,
    evaluate_baseline,
    load_dataset,
    parse_dataset,
)
from evaluation.claim_ranking.metrics import EvaluationInputError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _payload() -> dict[str, Any]:
    return {
        "dataset_id": "synthetic-fixture",
        "version": "1",
        "description": "Synthetic unit-test fixture only.",
        "labeling_guide_version": "1",
        "examples": [
            {
                "claim_id": "A",
                "claim": "Synthetic claim A.",
                "relevance_grade": 3,
                "rationale": "Synthetic rationale A.",
                "baseline_ranks": {"O1": 1, "O2": 2, "O3": None},
                "ecs": 0.5,
                "evidence": [
                    {
                        "title": "Source A",
                        "url": "https://example.test/a",
                        "content": "Synthetic evidence A.",
                        "retrieved_at": "2026-01-01",
                        "license": "CC0",
                    }
                ],
                "reproduction_notes": "Synthetic fixture.",
            },
            {
                "claim_id": "B",
                "claim": "Synthetic claim B.",
                "relevance_grade": 1,
                "rationale": "Synthetic rationale B.",
                "baseline_ranks": {"O1": 2, "O2": 1, "O3": 1},
                "ecs": None,
            },
            {
                "claim_id": "C",
                "claim": "Synthetic claim C.",
                "relevance_grade": 0,
                "rationale": "Synthetic rationale C.",
                "baseline_ranks": {"O1": 3, "O2": None, "O3": 2},
                "ecs": 0.0,
            },
        ],
        "exclusions": [
            {
                "claim_id": "X",
                "reason": "Synthetic exclusion.",
            }
        ],
    }


def _first_example(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["examples"][0]


def test_schema_file_is_valid_and_declares_required_fields() -> None:
    schema_path = (
        PROJECT_ROOT
        / "evaluation"
        / "claim_ranking"
        / "data"
        / "claim_ranking_schema_v1.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["type"] == "object"
    assert "examples" in schema["required"]
    assert set(schema["$defs"]["example"]["required"]) == {
        "claim_id",
        "claim",
        "relevance_grade",
        "rationale",
        "baseline_ranks",
        "ecs",
    }
    assert set(
        schema["$defs"]["example"]["properties"]["baseline_ranks"]["required"]
    ) == {"O1", "O2", "O3"}


def test_parse_dataset_valid_and_preserves_metadata() -> None:
    payload = _payload()
    payload["notes"] = "Synthetic metadata."

    dataset = parse_dataset(payload)

    assert dataset.path == Path("<memory>")
    assert dataset.sha256 == ""
    assert len(dataset.examples) == 3
    assert dataset.examples[0].claim_id == "A"
    assert dataset.examples[0].relevance_grade == 3
    assert dataset.examples[0].ecs == pytest.approx(0.5)
    assert dataset.examples[1].ecs is None
    assert dataset.examples[0].evidence[0].content == "Synthetic evidence A."
    assert dataset.examples[0].evidence[0].license == "CC0"
    assert dataset.exclusions[0].claim_id == "X"
    assert dataset.metadata["notes"] == "Synthetic metadata."


def test_load_dataset_computes_sha256(tmp_path: Path) -> None:
    dataset_path = tmp_path / "fixture.json"
    raw_text = json.dumps(_payload(), indent=2)
    raw_bytes = raw_text.encode("utf-8")
    dataset_path.write_bytes(raw_bytes)

    dataset = load_dataset(dataset_path)

    expected_hash = hashlib.sha256(raw_bytes).hexdigest()
    assert dataset.path == dataset_path
    assert dataset.sha256 == expected_hash
    assert len(dataset.examples) == 3


def test_duplicate_claim_ids_rejected() -> None:
    payload = _payload()
    payload["examples"][1]["claim_id"] = "A"

    with pytest.raises(DatasetValidationError, match="Duplicate claim_id"):
        parse_dataset(payload)


@pytest.mark.parametrize("grade", [-1, 4, True, 1.5, "3"])
def test_invalid_relevance_grade_rejected(grade: object) -> None:
    payload = _payload()
    _first_example(payload)["relevance_grade"] = grade

    with pytest.raises(DatasetValidationError):
        parse_dataset(payload)


@pytest.mark.parametrize("rank", [0, -1, True, 1.5, "1"])
def test_invalid_baseline_rank_rejected(rank: object) -> None:
    payload = _payload()
    _first_example(payload)["baseline_ranks"]["O1"] = rank

    with pytest.raises(DatasetValidationError):
        parse_dataset(payload)


def test_baseline_ranks_require_all_objectives() -> None:
    payload = _payload()
    _first_example(payload)["baseline_ranks"] = {"O1": 1, "O2": 2}

    with pytest.raises(DatasetValidationError, match="exactly O1, O2, O3"):
        parse_dataset(payload)


def test_baseline_ranks_reject_extra_objectives() -> None:
    payload = _payload()
    _first_example(payload)["baseline_ranks"] = {
        "O1": 1,
        "O2": 2,
        "O3": 3,
        "O4": 4,
    }

    with pytest.raises(DatasetValidationError, match="exactly O1, O2, O3"):
        parse_dataset(payload)


@pytest.mark.parametrize(
    "ecs",
    [-0.1, 1.1, True, "0.5", float("nan"), float("inf")],
)
def test_invalid_ecs_rejected(ecs: object) -> None:
    payload = _payload()
    _first_example(payload)["ecs"] = ecs

    with pytest.raises(DatasetValidationError):
        parse_dataset(payload)


def test_missing_ecs_is_preserved_as_none() -> None:
    dataset = parse_dataset(_payload())

    assert dataset.examples[1].ecs is None
    assert dataset.examples[1].ecs != 0.0


def test_empty_examples_rejected() -> None:
    payload = _payload()
    payload["examples"] = []

    with pytest.raises(DatasetValidationError, match="at least one example"):
        parse_dataset(payload)


@pytest.mark.parametrize(
    "field",
    ["dataset_id", "version", "description", "labeling_guide_version"],
)
def test_missing_metadata_rejected(field: str) -> None:
    payload = _payload()
    del payload[field]

    with pytest.raises(DatasetValidationError, match="missing required"):
        parse_dataset(payload)


def test_dataset_must_be_an_object() -> None:
    with pytest.raises(DatasetValidationError, match="JSON object"):
        parse_dataset([])


def test_examples_must_be_a_list() -> None:
    payload = _payload()
    payload["examples"] = {}

    with pytest.raises(DatasetValidationError, match="must be a list"):
        parse_dataset(payload)


def test_duplicate_exclusion_ids_rejected() -> None:
    payload = _payload()
    payload["exclusions"] = [
        {"claim_id": "X", "reason": "First."},
        {"claim_id": "X", "reason": "Second."},
    ]

    with pytest.raises(DatasetValidationError, match="Duplicate exclusion"):
        parse_dataset(payload)


def test_exclusion_cannot_also_be_an_example() -> None:
    payload = _payload()
    payload["exclusions"] = [
        {"claim_id": "A", "reason": "Contradictory exclusion."},
    ]

    with pytest.raises(DatasetValidationError, match="both examples and exclusions"):
        parse_dataset(payload)


def test_evidence_requires_content() -> None:
    payload = _payload()
    _first_example(payload)["evidence"] = [{"title": "No content"}]

    with pytest.raises(DatasetValidationError, match="content"):
        parse_dataset(payload)


def test_evidence_must_be_a_list() -> None:
    payload = _payload()
    _first_example(payload)["evidence"] = {}

    with pytest.raises(DatasetValidationError, match="must be a list"):
        parse_dataset(payload)


def test_baseline_ranking_per_objective_and_exclusions() -> None:
    dataset = parse_dataset(_payload())

    o1 = baseline_ranking(dataset, "O1")
    o2 = baseline_ranking(dataset, "O2")
    o3 = baseline_ranking(dataset, "O3")

    assert o1.ranked_claim_ids == ("A", "B", "C")
    assert o2.ranked_claim_ids == ("B", "A")
    assert o3.ranked_claim_ids == ("B", "C")

    assert o1.exclusions == ()
    assert [item.claim_id for item in o2.exclusions] == ["C"]
    assert [item.claim_id for item in o3.exclusions] == ["A"]


def test_baseline_ranking_duplicate_ranks_use_dataset_order() -> None:
    payload = _payload()
    _first_example(payload)["baseline_ranks"]["O1"] = 1
    payload["examples"][1]["baseline_ranks"]["O1"] = 1
    payload["examples"][2]["baseline_ranks"]["O1"] = 3

    dataset = parse_dataset(payload)
    o1 = baseline_ranking(dataset, "O1")

    assert o1.ranked_claim_ids == ("A", "B", "C")


def test_baseline_ranking_rejects_invalid_objective() -> None:
    dataset = parse_dataset(_payload())

    with pytest.raises(DatasetValidationError, match="objective must be"):
        baseline_ranking(dataset, "O4")


def test_dataset_summary_counts() -> None:
    dataset = parse_dataset(_payload())
    summary = dataset_summary(dataset)

    assert summary["num_examples"] == 3
    assert summary["relevance_grade_counts"] == {
        "0": 1,
        "1": 1,
        "2": 0,
        "3": 1,
    }
    assert summary["missing_ecs_count"] == 1
    assert summary["baseline_rank_counts"] == {
        "O1": 3,
        "O2": 2,
        "O3": 2,
    }
    assert summary["missing_baseline_rank_counts"] == {
        "O1": 0,
        "O2": 1,
        "O3": 1,
    }
    assert summary["dataset_exclusions"] == [
        {
            "claim_id": "X",
            "reason": "Synthetic exclusion.",
        }
    ]


def test_evaluate_baseline_returns_per_objective_metrics() -> None:
    dataset = parse_dataset(_payload())
    report = evaluate_baseline(dataset, [1, 2])

    assert report["k_values"] == [1, 2]
    assert report["relevant_min_grade"] == 1

    o1 = report["objectives"]["O1"]
    assert o1["num_claims"] == 3
    assert o1["num_ranked"] == 3
    assert o1["exclusions"] == []
    assert o1["metrics"]["1"]["precision"] == pytest.approx(1.0)
    assert o1["metrics"]["1"]["recall"] == pytest.approx(1 / 2)
    assert o1["metrics"]["1"]["ndcg"] == pytest.approx(1.0)
    assert o1["metrics"]["2"]["precision"] == pytest.approx(1.0)
    assert o1["metrics"]["2"]["recall"] == pytest.approx(1.0)

    o2 = report["objectives"]["O2"]
    assert o2["num_claims"] == 2
    assert o2["num_ranked"] == 2
    assert o2["exclusions"] == [
        {
            "claim_id": "C",
            "reason": "No baseline rank for O2.",
        }
    ]
    assert o2["metrics"]["1"]["precision"] == pytest.approx(1.0)
    assert o2["metrics"]["1"]["recall"] == pytest.approx(1 / 2)

    o3 = report["objectives"]["O3"]
    assert o3["num_claims"] == 2
    assert o3["exclusions"] == [
        {
            "claim_id": "A",
            "reason": "No baseline rank for O3.",
        }
    ]
    assert o3["metrics"]["2"]["precision"] == pytest.approx(1 / 2)
    assert o3["metrics"]["2"]["recall"] == pytest.approx(1.0)


def test_evaluate_baseline_handles_no_eligible_claims() -> None:
    payload = _payload()
    for example in payload["examples"]:
        example["baseline_ranks"] = {"O1": None, "O2": None, "O3": None}

    dataset = parse_dataset(payload)
    report = evaluate_baseline(dataset, [1])

    for objective in ("O1", "O2", "O3"):
        objective_report = report["objectives"][objective]
        assert objective_report["num_claims"] == 0
        assert objective_report["num_ranked"] == 0
        assert len(objective_report["exclusions"]) == 3
        assert objective_report["metrics"]["1"] == {
            "precision": None,
            "recall": None,
            "ndcg": None,
        }


def test_evaluate_baseline_rejects_invalid_k() -> None:
    dataset = parse_dataset(_payload())

    with pytest.raises(EvaluationInputError):
        evaluate_baseline(dataset, [0])
