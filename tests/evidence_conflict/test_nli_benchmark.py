"""Unit tests for the diagnostic NLI stance benchmark.

These tests use fake classifiers and in-memory datasets. They do not import
PyTorch or Transformers, download model weights, or call external services.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation.evidence_conflict.benchmark import (
    DEFAULT_DATASET_PATH,
    EXPECTED_CLASS_COUNTS,
    LABEL_VALUES,
    BenchmarkDataset,
    DatasetValidationError,
    build_report,
    class_counts,
    compute_metrics,
    compute_runtime,
    evaluate_examples,
    get_peak_working_set_bytes,
    load_dataset,
    parse_dataset,
    render_markdown,
    validate_dataset,
    write_reports,
)
from src.evidence_conflict import ECSConfig
from src.evidence_conflict.schemas import (
    StanceLabel,
    StancePrediction,
    StanceProbabilities,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _record(
    example_id: str,
    gold: str,
    predicted: str | None,
    *,
    correct: bool | None = None,
    error_type: str | None = None,
    category: str = "test",
    inference_seconds: float = 0.01,
) -> dict[str, object]:
    if correct is None:
        correct = predicted == gold
    return {
        "id": example_id,
        "claim": f"claim for {example_id}",
        "evidence": f"evidence for {example_id}",
        "gold_label": gold,
        "predicted_label": predicted,
        "confidence": 0.8 if predicted is not None else None,
        "probabilities": (
            {
                "supports": 0.8,
                "refutes": 0.1,
                "not_enough_info": 0.1,
            }
            if predicted is not None
            else None
        ),
        "correct": bool(correct),
        "abstained": False if predicted is not None else None,
        "truncated": False if predicted is not None else None,
        "evidence_index": 0 if predicted is not None else None,
        "inference_seconds": inference_seconds,
        "error_type": error_type,
        "error_message": (
            "synthetic failure" if error_type is not None else None
        ),
        "rationale": "test rationale",
        "category": category,
    }


def _prediction(
    label: StanceLabel,
    *,
    evidence_index: int = 0,
) -> StancePrediction:
    if label is StanceLabel.SUPPORTS:
        probabilities = StanceProbabilities(0.8, 0.1, 0.1)
    elif label is StanceLabel.REFUTES:
        probabilities = StanceProbabilities(0.1, 0.8, 0.1)
    else:
        probabilities = StanceProbabilities(0.1, 0.1, 0.8)
    return StancePrediction(
        evidence_index=evidence_index,
        label=label,
        confidence=0.8,
        probabilities=probabilities,
    )


class FakeClassifier:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def predict(self, claim: str, evidence: list[str]) -> list[StancePrediction]:
        self.calls.append((claim, tuple(evidence)))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return [outcome]  # type: ignore[list-item]


def _tiny_dataset_payload() -> dict[str, object]:
    return {
        "version": "test",
        "examples": [
            {
                "id": "SUP-T01",
                "claim": "A test claim that is supported.",
                "evidence": "Evidence that supports the test claim.",
                "gold_label": "supports",
                "rationale": "The evidence supports the claim.",
                "category": "test_support",
            },
            {
                "id": "REF-T01",
                "claim": "A test claim that is refuted.",
                "evidence": "Evidence that refutes the test claim.",
                "gold_label": "refutes",
                "rationale": "The evidence refutes the claim.",
                "category": "test_refute",
            },
            {
                "id": "NEI-T01",
                "claim": "A test claim with no decision.",
                "evidence": "Evidence that is related but not decisive.",
                "gold_label": "not_enough_info",
                "rationale": "The evidence is non-decisive.",
                "category": "test_nei",
            },
        ],
    }


def _tiny_dataset() -> BenchmarkDataset:
    return parse_dataset(_tiny_dataset_payload(), sha256="test-sha")


def _config() -> ECSConfig:
    return ECSConfig(
        model_name="fake-model",
        device="cpu",
        max_length=128,
        batch_size=1,
        local_files_only=True,
    )


def _raw_example(**overrides: object) -> dict[str, object]:
    example: dict[str, object] = {
        "id": "T-001",
        "claim": "A test claim.",
        "evidence": "A test evidence passage.",
        "gold_label": "supports",
        "rationale": "A test rationale.",
        "category": "test",
    }
    example.update(overrides)
    return example


def test_versioned_dataset_has_expected_counts_and_stable_ids() -> None:
    dataset = load_dataset(DEFAULT_DATASET_PATH)

    assert len(dataset.examples) == 90
    assert class_counts(dataset.examples) == {
        "supports": 30,
        "refutes": 30,
        "not_enough_info": 30,
    }
    assert validate_dataset(dataset, EXPECTED_CLASS_COUNTS) == {
        "supports": 30,
        "refutes": 30,
        "not_enough_info": 30,
    }
    assert len(dataset.sha256) == 64
    assert dataset.examples[0].example_id == "SUP-001"
    assert dataset.examples[-1].example_id == "NEI-030"

    expected_prefix = {
        "supports": "SUP-",
        "refutes": "REF-",
        "not_enough_info": "NEI-",
    }
    assert len({example.example_id for example in dataset.examples}) == 90
    for example in dataset.examples:
        assert example.example_id.startswith(
            expected_prefix[example.gold_label.value]
        )


def test_versioned_dataset_covers_required_diagnostic_categories() -> None:
    dataset = load_dataset(DEFAULT_DATASET_PATH)
    categories = {example.category for example in dataset.examples}

    assert {
        "direct_support",
        "direct_contradiction",
        "related_non_decisive",
        "unrelated",
        "overlapping_vocabulary",
        "numerical",
        "date",
        "named_entity",
        "location",
        "negation",
        "paraphrase",
        "definition",
    } <= categories


def test_metrics_accuracy_macro_f1_per_class_and_confusion_order() -> None:
    records = [
        _record("e1", "supports", "supports"),
        _record("e2", "supports", "refutes"),
        _record("e3", "refutes", "refutes"),
        _record("e4", "refutes", "refutes"),
        _record("e5", "not_enough_info", "not_enough_info"),
        _record("e6", "not_enough_info", "supports"),
    ]

    metrics = compute_metrics(records)

    assert metrics["total_examples"] == 6
    assert metrics["evaluated_examples"] == 6
    assert metrics["inference_failures"] == 0
    assert metrics["correct_examples"] == 4
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["accuracy_denominator"] == 6
    assert metrics["macro_f1_denominator"] == 6
    assert metrics["macro_f1"] == pytest.approx(
        (0.5 + 0.8 + (2 / 3)) / 3
    )
    assert metrics["confusion_matrix_labels"] == list(LABEL_VALUES)
    assert metrics["confusion_matrix"] == [
        [1, 1, 0],
        [0, 2, 0],
        [1, 0, 1],
    ]

    per_class = {item["label"]: item for item in metrics["per_class"]}
    assert per_class["supports"]["precision"] == pytest.approx(0.5)
    assert per_class["supports"]["recall"] == pytest.approx(0.5)
    assert per_class["supports"]["f1"] == pytest.approx(0.5)
    assert per_class["refutes"]["precision"] == pytest.approx(2 / 3)
    assert per_class["refutes"]["recall"] == pytest.approx(1.0)
    assert per_class["refutes"]["f1"] == pytest.approx(0.8)
    assert per_class["not_enough_info"]["precision"] == pytest.approx(1.0)
    assert per_class["not_enough_info"]["recall"] == pytest.approx(0.5)
    assert per_class["not_enough_info"]["f1"] == pytest.approx(2 / 3)
    assert [item["support"] for item in metrics["per_class"]] == [2, 2, 2]


def test_metrics_handle_missing_classes_with_fixed_label_set() -> None:
    records = [
        _record("e1", "supports", "supports"),
        _record("e2", "supports", "refutes"),
        _record("e3", "refutes", "refutes"),
    ]

    metrics = compute_metrics(records)

    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["confusion_matrix"] == [
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    assert metrics["macro_f1"] == pytest.approx(
        ((2 / 3) + (2 / 3) + 0.0) / 3
    )

    per_class = {item["label"]: item for item in metrics["per_class"]}
    assert per_class["not_enough_info"]["support"] == 0
    assert per_class["not_enough_info"]["gold_total"] == 0
    assert per_class["not_enough_info"]["precision"] == 0.0
    assert per_class["not_enough_info"]["recall"] == 0.0
    assert per_class["not_enough_info"]["f1"] == 0.0


def test_metrics_record_failures_separately_and_do_not_count_them_correct() -> None:
    records = [
        _record("e1", "supports", "supports"),
        _record(
            "e2",
            "refutes",
            None,
            correct=False,
            error_type="RuntimeError",
        ),
        _record("e3", "not_enough_info", "not_enough_info"),
    ]

    metrics = compute_metrics(records)

    assert metrics["total_examples"] == 3
    assert metrics["evaluated_examples"] == 2
    assert metrics["inference_failures"] == 1
    assert metrics["correct_examples"] == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["accuracy_denominator"] == 2
    assert metrics["macro_f1"] == pytest.approx(2 / 3)
    assert metrics["accuracy_including_failures_as_incorrect"] == pytest.approx(
        2 / 3
    )
    assert metrics["failure_aware_accuracy_denominator"] == 3
    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["failure_rate"] == pytest.approx(1 / 3)

    per_class = {item["label"]: item for item in metrics["per_class"]}
    assert per_class["refutes"]["support"] == 0
    assert per_class["refutes"]["gold_total"] == 1
    assert per_class["refutes"]["inference_failures"] == 1
    assert per_class["refutes"]["recall_including_failures"] == 0.0


def test_metrics_all_failures_return_undefined_metrics() -> None:
    records = [
        _record("e1", "supports", None, correct=False, error_type="Error"),
        _record("e2", "refutes", None, correct=False, error_type="Error"),
    ]

    metrics = compute_metrics(records)

    assert metrics["evaluated_examples"] == 0
    assert metrics["inference_failures"] == 2
    assert metrics["accuracy"] is None
    assert metrics["macro_f1"] is None
    assert metrics["accuracy_including_failures_as_incorrect"] == 0.0
    assert metrics["confusion_matrix"] == [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    assert all(item["precision"] is None for item in metrics["per_class"])
    assert all(item["recall"] is None for item in metrics["per_class"])
    assert all(item["f1"] is None for item in metrics["per_class"])


def test_compute_runtime_separates_initial_and_warm_calls() -> None:
    records = [
        _record("e1", "supports", "supports", inference_seconds=10.0),
        _record("e2", "supports", "supports", inference_seconds=1.0),
        _record(
            "e3",
            "refutes",
            None,
            correct=False,
            error_type="Error",
            inference_seconds=2.0,
        ),
        _record("e4", "refutes", "refutes", inference_seconds=3.0),
    ]

    runtime = compute_runtime(records)

    assert runtime["total_attempt_seconds"] == pytest.approx(16.0)
    assert runtime["initial_call_seconds"] == pytest.approx(10.0)
    assert runtime["warm_attempts"] == 3
    assert runtime["warm_inference_seconds"] == pytest.approx(6.0)
    assert runtime["average_warm_inference_seconds"] == pytest.approx(2.0)
    assert runtime["warm_successful_attempts"] == 2
    assert runtime["average_warm_successful_inference_seconds"] == pytest.approx(
        2.0
    )


def test_peak_working_set_measurement_is_sane() -> None:
    value = get_peak_working_set_bytes()

    assert value is None or value > 0


def test_evaluate_examples_preserves_order_and_uses_one_classifier() -> None:
    dataset = _tiny_dataset()
    classifier = FakeClassifier(
        [
            _prediction(StanceLabel.SUPPORTS),
            _prediction(StanceLabel.REFUTES),
            _prediction(StanceLabel.NOT_ENOUGH_INFO),
        ]
    )

    output = evaluate_examples(dataset, classifier, _config())

    assert [record["id"] for record in output.records] == [
        "SUP-T01",
        "REF-T01",
        "NEI-T01",
    ]
    assert [record["predicted_label"] for record in output.records] == [
        "supports",
        "refutes",
        "not_enough_info",
    ]
    assert all(record["correct"] for record in output.records)
    assert output.metrics["accuracy"] == 1.0
    assert output.metrics["inference_failures"] == 0
    assert len(classifier.calls) == 3
    assert classifier.calls[0] == (
        "A test claim that is supported.",
        ("Evidence that supports the test claim.",),
    )


def test_evaluate_examples_records_failure_and_continues() -> None:
    dataset = _tiny_dataset()
    classifier = FakeClassifier(
        [
            _prediction(StanceLabel.SUPPORTS),
            RuntimeError("synthetic inference failure"),
            _prediction(StanceLabel.NOT_ENOUGH_INFO),
        ]
    )

    output = evaluate_examples(dataset, classifier, _config())

    assert len(output.records) == 3
    failure = output.records[1]
    assert failure["id"] == "REF-T01"
    assert failure["predicted_label"] is None
    assert failure["correct"] is False
    assert failure["error_type"] == "RuntimeError"
    assert failure["error_message"] == "synthetic inference failure"
    assert output.metrics["total_examples"] == 3
    assert output.metrics["evaluated_examples"] == 2
    assert output.metrics["inference_failures"] == 1
    assert output.metrics["accuracy"] == 1.0
    assert output.metrics["accuracy_including_failures_as_incorrect"] == pytest.approx(
        2 / 3
    )


def test_evaluate_examples_treats_malformed_output_as_failure() -> None:
    class EmptyClassifier:
        def predict(self, claim: str, evidence: list[str]) -> list[StancePrediction]:
            return []

    output = evaluate_examples(_tiny_dataset(), EmptyClassifier(), _config())

    assert output.metrics["inference_failures"] == 3
    assert output.metrics["evaluated_examples"] == 0
    assert all(
        record["error_type"] == "UnexpectedPredictionCount"
        for record in output.records
    )


def test_build_and_write_reports_preserve_ids_and_include_failures(
    tmp_path: Path,
) -> None:
    dataset = _tiny_dataset()
    classifier = FakeClassifier(
        [
            _prediction(StanceLabel.SUPPORTS),
            _prediction(StanceLabel.SUPPORTS),
            RuntimeError("report failure"),
        ]
    )
    output = evaluate_examples(dataset, classifier, _config())
    report = build_report(
        dataset,
        output,
        _config(),
        environment={"python": "3.12", "packages": {"torch": "test"}},
        command="pytest",
        generated_at="2026-01-01T00:00:00+00:00",
    )

    json_path, markdown_path = write_reports(report, tmp_path)

    assert json_path.exists()
    assert markdown_path.exists()
    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert written["generated_at_utc"] == "2026-01-01T00:00:00+00:00"
    assert written["dataset"]["class_counts"] == {
        "supports": 1,
        "refutes": 1,
        "not_enough_info": 1,
    }
    assert [item["id"] for item in written["examples"]] == [
        "SUP-T01",
        "REF-T01",
        "NEI-T01",
    ]
    assert written["metrics"]["inference_failures"] == 1
    assert [item["id"] for item in written["inference_failures"]] == [
        "NEI-T01"
    ]
    assert [item["id"] for item in written["misclassified_examples"]] == [
        "REF-T01"
    ]

    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown.startswith("# NLI Stance Diagnostic Benchmark")
    assert "## Confusion matrix" in markdown
    assert "## Inference failures" in markdown
    assert "## Misclassified examples" in markdown
    assert "SUP-T01" in markdown
    assert "REF-T01" in markdown
    assert "NEI-T01" in markdown
    assert "FAILED" in markdown
    assert "report failure" in markdown


def test_render_markdown_handles_no_failures_and_no_misclassifications() -> None:
    dataset = _tiny_dataset()
    classifier = FakeClassifier(
        [
            _prediction(StanceLabel.SUPPORTS),
            _prediction(StanceLabel.REFUTES),
            _prediction(StanceLabel.NOT_ENOUGH_INFO),
        ]
    )
    output = evaluate_examples(dataset, classifier, _config())
    report = build_report(
        dataset,
        output,
        _config(),
        generated_at="2026-01-01T00:00:00+00:00",
    )

    markdown = render_markdown(report)

    assert "## Inference failures\n\nNone." in markdown
    assert "## Misclassified examples\n\nNone." in markdown
    assert "supports | refutes | not_enough_info" in markdown


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (
            {"examples": [{"id": "X-001"}]},
            "missing required field",
        ),
        (
            {"examples": [{"id": "X-001", "claim": "c", "evidence": "e",
                           "gold_label": "maybe", "rationale": "r",
                           "category": "c"}]},
            "must be one of",
        ),
        (
            {"examples": [{**_raw_example(), "claim": "   "}]},
            "claim must be a non-empty string",
        ),
        (
            {"examples": [{**_raw_example(), "evidence": ""}]},
            "evidence must be a non-empty string",
        ),
        (
            {"examples": [_raw_example(), _raw_example()]},
            "Duplicate example id",
        ),
        (
            {"examples": "not-a-list"},
            "examples",
        ),
        (
            {"examples": []},
            "at least one example",
        ),
        (
            {"examples": ["not-an-object"]},
            "must be an object",
        ),
        (
            {"not_examples": []},
            "examples",
        ),
    ],
)
def test_malformed_dataset_records_are_rejected(
    payload: object,
    match: str,
) -> None:
    with pytest.raises(DatasetValidationError, match=match):
        parse_dataset(payload)


def test_dataset_accepts_top_level_list_and_reports_metadata_empty() -> None:
    dataset = parse_dataset([_raw_example()])

    assert dataset.metadata == {}
    assert len(dataset.examples) == 1
    assert dataset.examples[0].example_id == "T-001"


def test_validate_dataset_rejects_unexpected_class_counts() -> None:
    dataset = parse_dataset({"examples": [_raw_example()]})

    with pytest.raises(DatasetValidationError, match="class counts"):
        validate_dataset(
            dataset,
            expected_counts={
                "supports": 2,
                "refutes": 0,
                "not_enough_info": 0,
            },
        )


def test_importing_benchmark_module_does_not_load_heavy_backends() -> None:
    code = "\n".join(
        [
            "import sys",
            "import evaluation.evidence_conflict.benchmark",
            "assert 'torch' not in sys.modules, 'torch was imported'",
            "assert 'transformers' not in sys.modules, "
            "'transformers was imported'",
            "print('ok')",
        ]
    )

    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
