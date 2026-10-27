"""Reproducible diagnostic benchmark for the NLI stance classifier.

This module intentionally uses the existing ``NLIStanceClassifier`` public API.
It does not modify, wrap, or reimplement the classifier's label mapping.

Ordinary unit tests can exercise this module with fake classifiers. Running the
module as a script performs real local inference and may load the configured
model from the Hugging Face cache.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.evidence_conflict import ECSConfig, StanceLabel
from src.evidence_conflict.nli_classifier import NLIStanceClassifier
from src.evidence_conflict.schemas import StancePrediction

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent / "data" / "nli_stance_benchmark_v1.json"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"

LABEL_ORDER: tuple[StanceLabel, ...] = (
    StanceLabel.SUPPORTS,
    StanceLabel.REFUTES,
    StanceLabel.NOT_ENOUGH_INFO,
)
LABEL_VALUES: tuple[str, ...] = tuple(label.value for label in LABEL_ORDER)
EXPECTED_CLASS_COUNTS: dict[StanceLabel, int] = {
    StanceLabel.SUPPORTS: 30,
    StanceLabel.REFUTES: 30,
    StanceLabel.NOT_ENOUGH_INFO: 30,
}


class DatasetValidationError(ValueError):
    """Raised when the benchmark dataset is malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class BenchmarkExample:
    """One manually labelled claim-evidence pair."""

    example_id: str
    claim: str
    evidence: str
    gold_label: StanceLabel
    rationale: str
    category: str


@dataclass(frozen=True, slots=True)
class BenchmarkDataset:
    """Validated benchmark examples plus file provenance."""

    path: Path
    sha256: str
    metadata: Mapping[str, Any]
    examples: tuple[BenchmarkExample, ...]


@dataclass(frozen=True, slots=True)
class EvaluationOutput:
    """Per-example records, metrics, and runtime information."""

    records: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    runtime: dict[str, Any]
    resolved_device: str | None
    model_class: str | None
    model_dtype: str | None
    tokenizer_class: str | None


class ClassifierProtocol(Protocol):
    """Minimal classifier interface used by the evaluator."""

    def predict(
        self,
        claim: str,
        evidence: Sequence[str],
    ) -> list[StancePrediction]:
        """Return one stance prediction per evidence passage."""


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{field} must be a non-empty string.")
    return value.strip()


def _coerce_label(value: object, field: str) -> StanceLabel:
    if isinstance(value, StanceLabel):
        return value
    if isinstance(value, str) and value in LABEL_VALUES:
        return StanceLabel(value)
    raise DatasetValidationError(
        f"{field} must be one of {LABEL_VALUES}; got {value!r}."
    )


def _parse_example(raw: object, index: int, seen_ids: set[str]) -> BenchmarkExample:
    if not isinstance(raw, Mapping):
        raise DatasetValidationError(
            f"examples[{index}] must be an object; got {type(raw).__name__}."
        )

    required = ("id", "claim", "evidence", "gold_label", "rationale", "category")
    missing = [field for field in required if field not in raw]
    if missing:
        raise DatasetValidationError(
            f"examples[{index}] is missing required field(s): {', '.join(missing)}."
        )

    example_id = _nonempty_string(raw["id"], f"examples[{index}].id")
    if example_id in seen_ids:
        raise DatasetValidationError(f"Duplicate example id {example_id!r}.")
    seen_ids.add(example_id)

    return BenchmarkExample(
        example_id=example_id,
        claim=_nonempty_string(raw["claim"], f"examples[{index}].claim"),
        evidence=_nonempty_string(
            raw["evidence"],
            f"examples[{index}].evidence",
        ),
        gold_label=_coerce_label(
            raw["gold_label"],
            f"examples[{index}].gold_label",
        ),
        rationale=_nonempty_string(
            raw["rationale"],
            f"examples[{index}].rationale",
        ),
        category=_nonempty_string(
            raw["category"],
            f"examples[{index}].category",
        ),
    )


def _parse_payload(payload: object) -> tuple[tuple[BenchmarkExample, ...], dict[str, Any]]:
    if isinstance(payload, Mapping):
        metadata = dict(payload)
        raw_examples = metadata.pop("examples", None)
    else:
        metadata = {}
        raw_examples = payload

    if not isinstance(raw_examples, list):
        raise DatasetValidationError(
            "Dataset must be a JSON object with an 'examples' list, "
            "or a JSON list of examples."
        )
    if not raw_examples:
        raise DatasetValidationError("Dataset must contain at least one example.")

    seen_ids: set[str] = set()
    examples = tuple(
        _parse_example(raw, index, seen_ids)
        for index, raw in enumerate(raw_examples)
    )
    return examples, metadata


def parse_dataset(
    payload: object,
    *,
    path: Path = Path("<memory>"),
    sha256: str = "",
) -> BenchmarkDataset:
    """Parse and validate an in-memory dataset payload."""

    examples, metadata = _parse_payload(payload)
    return BenchmarkDataset(
        path=Path(path),
        sha256=sha256,
        metadata=metadata,
        examples=examples,
    )


def load_dataset(path: str | Path) -> BenchmarkDataset:
    """Load and validate a JSON benchmark dataset from disk."""

    dataset_path = Path(path)
    try:
        raw_bytes = dataset_path.read_bytes()
    except OSError as exc:
        raise DatasetValidationError(
            f"Could not read dataset {dataset_path}: {exc}"
        ) from exc

    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(
            f"Dataset {dataset_path} is not valid UTF-8 JSON: {exc}"
        ) from exc

    examples, metadata = _parse_payload(payload)
    return BenchmarkDataset(
        path=dataset_path,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        metadata=metadata,
        examples=examples,
    )


def class_counts(
    examples: Iterable[BenchmarkExample],
) -> dict[str, int]:
    """Return counts in the canonical label order."""

    counts = Counter(example.gold_label for example in examples)
    return {
        label.value: int(counts.get(label, 0))
        for label in LABEL_ORDER
    }


def validate_dataset(
    dataset: BenchmarkDataset,
    expected_counts: Mapping[StanceLabel | str, int] | None = None,
) -> dict[str, int]:
    """Validate class counts and return the observed counts."""

    counts = Counter(example.gold_label for example in dataset.examples)
    observed = {
        label.value: int(counts.get(label, 0))
        for label in LABEL_ORDER
    }

    if expected_counts is not None:
        expected: dict[StanceLabel, int] = {}
        for raw_label, raw_count in expected_counts.items():
            label = _coerce_label(raw_label, "expected_class_counts key")
            if isinstance(raw_count, bool) or not isinstance(raw_count, int):
                raise DatasetValidationError(
                    f"Expected count for {label.value!r} must be an integer."
                )
            if raw_count < 0:
                raise DatasetValidationError(
                    f"Expected count for {label.value!r} must be >= 0."
                )
            expected[label] = raw_count

        actual_by_label = {
            label: int(counts.get(label, 0))
            for label in LABEL_ORDER
        }
        if actual_by_label != expected:
            raise DatasetValidationError(
                "Dataset class counts do not match expected counts: "
                f"expected {expected}, got {actual_by_label}."
            )

    return observed


def _failure_record(
    example: BenchmarkExample,
    elapsed: float,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "id": example.example_id,
        "claim": example.claim,
        "evidence": example.evidence,
        "gold_label": example.gold_label.value,
        "predicted_label": None,
        "confidence": None,
        "probabilities": None,
        "correct": False,
        "abstained": None,
        "truncated": None,
        "evidence_index": None,
        "inference_seconds": elapsed,
        "error_type": error_type,
        "error_message": error_message,
        "rationale": example.rationale,
        "category": example.category,
    }


def _success_record(
    example: BenchmarkExample,
    prediction: StancePrediction,
    elapsed: float,
) -> dict[str, Any]:
    probabilities = prediction.probabilities
    return {
        "id": example.example_id,
        "claim": example.claim,
        "evidence": example.evidence,
        "gold_label": example.gold_label.value,
        "predicted_label": prediction.label.value,
        "confidence": prediction.confidence,
        "probabilities": {
            "supports": probabilities.supports,
            "refutes": probabilities.refutes,
            "not_enough_info": probabilities.not_enough_info,
        },
        "correct": prediction.label is example.gold_label,
        "abstained": prediction.abstained,
        "truncated": prediction.truncated,
        "evidence_index": prediction.evidence_index,
        "inference_seconds": elapsed,
        "error_type": None,
        "error_message": None,
        "rationale": example.rationale,
        "category": example.category,
    }


def evaluate_examples(
    dataset: BenchmarkDataset,
    classifier: ClassifierProtocol,
    config: ECSConfig,
) -> EvaluationOutput:
    """Run one classifier call per example and collect complete records."""

    records: list[dict[str, Any]] = []

    for example in dataset.examples:
        started = time.perf_counter()
        try:
            raw_predictions = classifier.predict(
                example.claim,
                [example.evidence],
            )
            predictions = list(raw_predictions)
        except Exception as exc:  # noqa: BLE001 - failures are data, not control flow
            elapsed = time.perf_counter() - started
            records.append(
                _failure_record(
                    example,
                    elapsed,
                    type(exc).__name__,
                    str(exc),
                )
            )
            continue

        elapsed = time.perf_counter() - started
        if len(predictions) != 1:
            records.append(
                _failure_record(
                    example,
                    elapsed,
                    "UnexpectedPredictionCount",
                    f"Expected 1 prediction, got {len(predictions)}.",
                )
            )
            continue

        prediction = predictions[0]
        if not isinstance(prediction, StancePrediction):
            records.append(
                _failure_record(
                    example,
                    elapsed,
                    "UnexpectedPredictionType",
                    "Classifier returned "
                    f"{type(prediction).__name__}, expected StancePrediction.",
                )
            )
            continue

        if prediction.evidence_index != 0:
            records.append(
                _failure_record(
                    example,
                    elapsed,
                    "UnexpectedEvidenceIndex",
                    "Expected evidence index 0, got "
                    f"{prediction.evidence_index}.",
                )
            )
            continue

        records.append(_success_record(example, prediction, elapsed))

    metrics = compute_metrics(records)
    runtime = compute_runtime(records)
    model = getattr(classifier, "_model", None)
    tokenizer = getattr(classifier, "_tokenizer", None)
    model_dtype = getattr(model, "dtype", None)

    return EvaluationOutput(
        records=tuple(records),
        metrics=metrics,
        runtime=runtime,
        resolved_device=getattr(classifier, "_device", None),
        model_class=type(model).__name__ if model is not None else None,
        model_dtype=str(model_dtype) if model_dtype is not None else None,
        tokenizer_class=type(tokenizer).__name__ if tokenizer is not None else None,
    )


def _per_class_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    class_counts_all: Counter[str],
    class_failures: Counter[str],
) -> list[dict[str, Any]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_VALUES,
        average=None,
        zero_division=0,
    )
    correct_by_class = Counter(
        record["gold_label"]
        for record in records
        if record["error_type"] is None and record["correct"]
    )

    result: list[dict[str, Any]] = []
    for index, label in enumerate(LABEL_VALUES):
        gold_total = int(class_counts_all.get(label, 0))
        correct = int(correct_by_class.get(label, 0))
        result.append(
            {
                "label": label,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
                "gold_total": gold_total,
                "correct": correct,
                "inference_failures": int(class_failures.get(label, 0)),
                "recall_including_failures": (
                    correct / gold_total if gold_total else None
                ),
            }
        )
    return result


def compute_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute metrics over successful predictions only.

    Inference failures are excluded from accuracy, macro-F1, and per-class
    precision/recall/F1, but are reported separately and also affect the
    failure-aware accuracy and per-class recall-including-failures values.
    """

    all_records = list(records)
    total = len(all_records)
    successful = [
        record
        for record in all_records
        if record.get("error_type") is None
        and record.get("predicted_label") is not None
    ]
    failed = [
        record
        for record in all_records
        if record.get("error_type") is not None
    ]

    y_true = [str(record["gold_label"]) for record in successful]
    y_pred = [str(record["predicted_label"]) for record in successful]
    class_counts_all: Counter[str] = Counter(
        str(record["gold_label"]) for record in all_records
    )
    class_failures: Counter[str] = Counter(
        str(record["gold_label"]) for record in failed
    )
    correct = sum(1 for record in successful if record["correct"])

    if successful:
        accuracy: float | None = float(accuracy_score(y_true, y_pred))
        macro_f1: float | None = float(
            f1_score(
                y_true,
                y_pred,
                labels=LABEL_VALUES,
                average="macro",
                zero_division=0,
            )
        )
        per_class = _per_class_metrics(
            y_true,
            y_pred,
            successful,
            class_counts_all,
            class_failures,
        )
        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=LABEL_VALUES,
        ).tolist()
    else:
        accuracy = None
        macro_f1 = None
        per_class = [
            {
                "label": label,
                "precision": None,
                "recall": None,
                "f1": None,
                "support": 0,
                "gold_total": int(class_counts_all.get(label, 0)),
                "correct": 0,
                "inference_failures": int(class_failures.get(label, 0)),
                "recall_including_failures": (
                    0.0
                    if class_counts_all.get(label, 0)
                    else None
                ),
            }
            for label in LABEL_VALUES
        ]
        matrix = [[0 for _ in LABEL_VALUES] for _ in LABEL_VALUES]

    confusions = [
        {
            "gold_label": LABEL_VALUES[gold_index],
            "predicted_label": LABEL_VALUES[predicted_index],
            "count": int(matrix[gold_index][predicted_index]),
        }
        for gold_index in range(len(LABEL_VALUES))
        for predicted_index in range(len(LABEL_VALUES))
        if gold_index != predicted_index
        and matrix[gold_index][predicted_index] > 0
    ]
    confusions.sort(key=lambda item: -item["count"])

    return {
        "total_examples": total,
        "evaluated_examples": len(successful),
        "inference_failures": len(failed),
        "correct_examples": correct,
        "coverage": len(successful) / total if total else None,
        "failure_rate": len(failed) / total if total else None,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "accuracy_including_failures_as_incorrect": (
            correct / total if total else None
        ),
        "metrics_denominator": "successful_predictions_only",
        "accuracy_denominator": len(successful),
        "macro_f1_denominator": len(successful),
        "per_class_denominator": len(successful),
        "failure_aware_accuracy_denominator": total,
        "macro_f1_label_set": list(LABEL_VALUES),
        "macro_f1_missing_class_policy": (
            "fixed_three_label_set_with_zero_division"
        ),
        "confusion_matrix": matrix,
        "confusion_matrix_labels": list(LABEL_VALUES),
        "confusion_matrix_row_order": "gold",
        "confusion_matrix_column_order": "predicted",
        "per_class": per_class,
        "confusions": confusions,
    }


def compute_runtime(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize initial-call and warm inference timing."""

    attempts = list(records)
    if not attempts:
        return {
            "total_attempt_seconds": None,
            "initial_call_seconds": None,
            "initial_call_includes_lazy_model_loading": True,
            "warm_attempts": 0,
            "warm_inference_seconds": None,
            "average_warm_inference_seconds": None,
            "warm_successful_attempts": 0,
            "average_warm_successful_inference_seconds": None,
        }

    initial = float(attempts[0]["inference_seconds"])
    warm = attempts[1:]
    warm_successful = [
        record for record in warm if record["error_type"] is None
    ]
    warm_seconds = sum(float(record["inference_seconds"]) for record in warm)
    warm_successful_seconds = sum(
        float(record["inference_seconds"]) for record in warm_successful
    )

    return {
        "total_attempt_seconds": sum(
            float(record["inference_seconds"]) for record in attempts
        ),
        "initial_call_seconds": initial,
        "initial_call_includes_lazy_model_loading": True,
        "warm_attempts": len(warm),
        "warm_inference_seconds": warm_seconds,
        "average_warm_inference_seconds": (
            warm_seconds / len(warm) if warm else None
        ),
        "warm_successful_attempts": len(warm_successful),
        "average_warm_successful_inference_seconds": (
            warm_successful_seconds / len(warm_successful)
            if warm_successful
            else None
        ),
    }


def build_report(
    dataset: BenchmarkDataset,
    evaluation: EvaluationOutput,
    config: ECSConfig,
    *,
    environment: Mapping[str, Any] | None = None,
    command: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the machine-readable report payload."""

    counts = Counter(example.gold_label for example in dataset.examples)
    categories = Counter(example.category for example in dataset.examples)
    records = list(evaluation.records)
    failures = [
        record for record in records if record["error_type"] is not None
    ]
    misclassified = [
        record
        for record in records
        if record["error_type"] is None and not record["correct"]
    ]

    metadata = {
        key: value
        for key, value in dataset.metadata.items()
        if key != "examples"
    }

    return {
        "report_schema_version": "1.0",
        "generated_at_utc": generated_at
        or datetime.now(timezone.utc).isoformat(),
        "benchmark": {
            "name": "nli_stance_diagnostic_benchmark",
            "purpose": (
                "Diagnostic evaluation of three-class NLI stance classification "
                "before ECS scoring."
            ),
            "scored_examples": len(dataset.examples),
        },
        "dataset": {
            "path": str(dataset.path),
            "sha256": dataset.sha256,
            "metadata": metadata,
            "total_examples": len(dataset.examples),
            "class_counts": {
                label.value: int(counts.get(label, 0))
                for label in LABEL_ORDER
            },
            "category_counts": dict(sorted(categories.items())),
        },
        "model": {
            "name": config.model_name,
            "configured_revision": config.model_revision,
            "resolved_revision": None,
            "resolved_revision_note": (
                "The current classifier API does not expose the resolved "
                "Hugging Face revision."
            ),
            "device_requested": config.device,
            "resolved_device": evaluation.resolved_device,
            "sequence_length": config.max_length,
            "batch_size": config.batch_size,
            "evidence_passages_per_call": 1,
            "effective_batch_size": 1,
            "batch_size_note": (
                "The benchmark calls predict once per example with one evidence "
                "passage, so the effective batch size is 1; the configured batch "
                "size only caps batching."
            ),
            "confidence_threshold": config.confidence_threshold,
            "max_evidence_chars": config.max_evidence_chars,
            "local_files_only": config.local_files_only,
            "model_class": evaluation.model_class,
            "model_dtype": evaluation.model_dtype,
            "tokenizer_class": evaluation.tokenizer_class,
        },
        "environment": dict(environment or {}),
        "runtime": {
            **evaluation.runtime,
            "single_classifier_instance": True,
            "model_loading_note": (
                "The first predict call lazily imports/loads the tokenizer and "
                "model; its duration includes loading plus one inference. "
                "Warm timings exclude that first call."
            ),
        },
        "metrics": evaluation.metrics,
        "examples": records,
        "inference_failures": failures,
        "misclassified_examples": misclassified,
        "reproducibility": {
            "command": command,
            "dataset_path": str(dataset.path),
            "dataset_sha256": dataset.sha256,
        },
    }


def _format_metric(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(_md_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_md_cell(cell) for cell in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a readable Markdown report from the JSON report payload."""

    dataset = report["dataset"]
    metrics = report["metrics"]
    model = report["model"]
    runtime = report["runtime"]
    metadata = dataset.get("metadata", {})
    limitations = metadata.get(
        "limitations",
        "This is a small diagnostic dataset, not a representative benchmark.",
    )
    review_policy = metadata.get("review_policy", "Not documented.")

    lines: list[str] = [
        "# NLI Stance Diagnostic Benchmark",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Scope and limitations",
        "",
        str(limitations),
        "",
        "The benchmark is a diagnostic integration check. It does not establish "
        "general factual reliability, and the gold labels are not model-generated.",
        "",
        "## Dataset",
        "",
        _md_table(
            ("Property", "Value"),
            (
                ("Path", dataset["path"]),
                ("SHA-256", dataset["sha256"]),
                ("Version", metadata.get("version", "n/a")),
                ("Total examples", dataset["total_examples"]),
                ("Label definition", json.dumps(metadata.get("labels", {}))),
                ("Review policy", review_policy),
            ),
        ),
        "",
        "### Class counts",
        "",
        _md_table(
            ("Label", "Count"),
            [
                (label, dataset["class_counts"].get(label, 0))
                for label in LABEL_VALUES
            ],
        ),
        "",
        "### Category counts",
        "",
        _md_table(
            ("Category", "Count"),
            sorted(dataset["category_counts"].items()),
        ),
        "",
        "## Model and configuration",
        "",
        _md_table(
            ("Property", "Value"),
            (
                ("Model", model["name"]),
                ("Configured revision", model["configured_revision"]),
                ("Resolved revision", model["resolved_revision"]),
                ("Device requested", model["device_requested"]),
                ("Resolved device", model["resolved_device"]),
                ("Sequence length", model["sequence_length"]),
                ("Configured batch size", model["batch_size"]),
                ("Evidence passages per call", model["evidence_passages_per_call"]),
                ("Effective batch size", model["effective_batch_size"]),
                ("Confidence threshold", model["confidence_threshold"]),
                ("Max evidence characters", model["max_evidence_chars"]),
                ("Local files only", model["local_files_only"]),
                ("Model class", model["model_class"]),
                ("Model dtype", model["model_dtype"]),
                ("Tokenizer class", model["tokenizer_class"]),
            ),
        ),
        "",
        "## Runtime",
        "",
        _md_table(
            ("Metric", "Value"),
            (
                ("Total attempt seconds", _format_metric(runtime.get("total_attempt_seconds"))),
                ("Initial call seconds (includes lazy loading)", _format_metric(runtime.get("initial_call_seconds"))),
                ("Warm attempts", runtime.get("warm_attempts")),
                ("Warm inference seconds", _format_metric(runtime.get("warm_inference_seconds"))),
                ("Average warm inference seconds", _format_metric(runtime.get("average_warm_inference_seconds"))),
                ("Warm successful attempts", runtime.get("warm_successful_attempts")),
                ("Average warm successful inference seconds", _format_metric(runtime.get("average_warm_successful_inference_seconds"))),
                ("Single classifier instance", runtime.get("single_classifier_instance")),
            ),
        ),
        "",
        "## Metrics",
        "",
        _md_table(
            ("Metric", "Value"),
            (
                ("Total examples", metrics["total_examples"]),
                ("Evaluated examples", metrics["evaluated_examples"]),
                ("Inference failures", metrics["inference_failures"]),
                ("Correct examples", metrics["correct_examples"]),
                ("Coverage", _format_metric(metrics["coverage"])),
                ("Failure rate", _format_metric(metrics["failure_rate"])),
                ("Accuracy", _format_metric(metrics["accuracy"])),
                ("Macro-F1", _format_metric(metrics["macro_f1"])),
                (
                    "Accuracy including failures as incorrect",
                    _format_metric(
                        metrics["accuracy_including_failures_as_incorrect"]
                    ),
                ),
                ("Metrics denominator", metrics["metrics_denominator"]),
            ),
        ),
        "",
        f"Accuracy and macro-F1 use {metrics['accuracy_denominator']} successful "
        "predictions. The failure-aware accuracy uses "
        f"{metrics['failure_aware_accuracy_denominator']} total examples. "
        "Macro-F1 uses the fixed three-label set; absent classes contribute 0.",
        "",
        "## Per-class metrics",
        "",
        _md_table(
            (
                "Label",
                "Precision",
                "Recall",
                "F1",
                "Support (evaluated)",
                "Gold total",
                "Failures",
                "Recall incl. failures",
            ),
            [
                (
                    item["label"],
                    _format_metric(item["precision"]),
                    _format_metric(item["recall"]),
                    _format_metric(item["f1"]),
                    item["support"],
                    item["gold_total"],
                    item["inference_failures"],
                    _format_metric(item["recall_including_failures"]),
                )
                for item in metrics["per_class"]
            ],
        ),
        "",
        "## Confusion matrix",
        "",
        "Rows are gold labels; columns are predicted labels.",
        "",
        _md_table(
            ("Gold \\ Predicted", *metrics["confusion_matrix_labels"]),
            [
                (
                    label,
                    *metrics["confusion_matrix"][index],
                )
                for index, label in enumerate(metrics["confusion_matrix_labels"])
            ],
        ),
        "",
        "## Most frequent confusion patterns",
        "",
    ]

    if metrics["confusions"]:
        lines.append(
            _md_table(
                ("Gold", "Predicted", "Count"),
                [
                    (
                        item["gold_label"],
                        item["predicted_label"],
                        item["count"],
                    )
                    for item in metrics["confusions"]
                ],
            )
        )
    else:
        lines.append("None.")

    lines.extend(
        [
            "",
            "## Inference failures",
            "",
        ]
    )
    failures = report["inference_failures"]
    if failures:
        for item in failures:
            lines.extend(
                [
                    f"- `{item['id']}` (gold: `{item['gold_label']}`): "
                    f"`{item['error_type']}`: {item['error_message']}",
                ]
            )
    else:
        lines.append("None.")

    lines.extend(
        [
            "",
            "## Misclassified examples",
            "",
        ]
    )
    misclassified = report["misclassified_examples"]
    if misclassified:
        for item in misclassified:
            lines.extend(
                [
                    f"- `{item['id']}` gold=`{item['gold_label']}` "
                    f"predicted=`{item['predicted_label']}` "
                    f"confidence={_format_metric(item['confidence'])} "
                    f"category=`{item['category']}`",
                    f"  - Claim: {item['claim']}",
                    f"  - Evidence: {item['evidence']}",
                    f"  - Rationale: {item['rationale']}",
                ]
            )
    else:
        lines.append("None.")

    lines.extend(
        [
            "",
            "## All predictions",
            "",
            _md_table(
                ("ID", "Gold", "Predicted", "Confidence", "Correct", "Category"),
                [
                    (
                        item["id"],
                        item["gold_label"],
                        item["predicted_label"] or "FAILED",
                        _format_metric(item["confidence"]),
                        item["correct"],
                        item["category"],
                    )
                    for item in report["examples"]
                ],
            ),
            "",
            "## Reproducibility",
            "",
            _md_table(
                ("Property", "Value"),
                (
                    ("Command", report["reproducibility"].get("command")),
                    ("Dataset path", report["reproducibility"].get("dataset_path")),
                    ("Dataset SHA-256", report["reproducibility"].get("dataset_sha256")),
                    ("Python", report.get("environment", {}).get("python")),
                    ("Platform", report.get("environment", {}).get("platform")),
                    ("PyTorch", report.get("environment", {}).get("packages", {}).get("torch")),
                    ("Transformers", report.get("environment", {}).get("packages", {}).get("transformers")),
                    ("CUDA available", report.get("environment", {}).get("cuda_available")),
                    ("Peak working set (MB)", _format_metric(report.get("environment", {}).get("peak_working_set_mb"), 1)),
                    ("Peak memory measurement", report.get("environment", {}).get("peak_working_set_measurement")),
                ),
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    report: Mapping[str, Any],
    output_dir: str | Path,
    *,
    prefix: str = "nli_stance_benchmark",
) -> tuple[Path, Path]:
    """Write JSON and Markdown reports; return their paths."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / f"{prefix}.json"
    markdown_path = destination / f"{prefix}.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def collect_environment() -> dict[str, Any]:
    """Collect lightweight environment metadata for the report."""

    package_names = (
        "torch",
        "transformers",
        "tokenizers",
        "scikit-learn",
        "numpy",
    )
    packages: dict[str, str | None] = {}
    for package in package_names:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    cuda_available: bool | None = None
    cuda_device_count: int | None = None
    cuda_device_name: str | None = None
    cuda_error: str | None = None
    try:
        import torch
    except Exception as exc:  # pragma: no cover - real environment dependent
        cuda_error = f"{type(exc).__name__}: {exc}"
    else:
        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = int(torch.cuda.device_count())
        if cuda_available:
            cuda_device_name = torch.cuda.get_device_name(0)

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_name": cuda_device_name,
        "cuda_error": cuda_error,
    }


def get_peak_working_set_bytes() -> int | None:
    """Return this process's peak working set on Windows, when available.

    This uses the Windows ``GetProcessMemoryInfo`` API through ``ctypes``.
    It requires no third-party package and is only a process-level peak, not a
    separate measurement of model weights alone.
    """

    if sys.platform != "win32":
        return None

    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        process = kernel32.GetCurrentProcess()
        success = psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if not success:
            return None
        peak = int(counters.PeakWorkingSetSize)
        return peak if peak > 0 else None
    except Exception:  # pragma: no cover - platform/API dependent
        return None


def _build_config(args: argparse.Namespace) -> ECSConfig:
    kwargs: dict[str, Any] = {}
    if args.model is not None:
        kwargs["model_name"] = args.model
    if args.device is not None:
        kwargs["device"] = args.device
    if args.max_length is not None:
        kwargs["max_length"] = args.max_length
    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    if args.confidence_threshold is not None:
        kwargs["confidence_threshold"] = args.confidence_threshold
    if args.model_revision is not None:
        kwargs["model_revision"] = args.model_revision
    if args.local_files_only:
        kwargs["local_files_only"] = True
    return ECSConfig(**kwargs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostic NLI stance benchmark against a local or "
            "downloadable model checkpoint."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the JSON benchmark dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON and Markdown reports.",
    )
    parser.add_argument(
        "--output-prefix",
        default="nli_stance_benchmark",
        help="Base filename for reports.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override; defaults to ECSConfig's model.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override; defaults to ECSConfig's auto device.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional token sequence length override.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch-size override.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Optional confidence-threshold override.",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Optional model revision override.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not contact the Hugging Face Hub; use cached files only.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        dataset = load_dataset(args.dataset)
        expected = dataset.metadata.get("expected_class_counts")
        validate_dataset(
            dataset,
            expected_counts=expected if isinstance(expected, Mapping) else None,
        )
    except DatasetValidationError as exc:
        print(f"Dataset validation failed: {exc}", file=sys.stderr)
        return 2

    config = _build_config(args)
    classifier = NLIStanceClassifier(config)
    evaluation = evaluate_examples(dataset, classifier, config)
    environment = collect_environment()
    peak_working_set_bytes = get_peak_working_set_bytes()
    environment["peak_working_set_bytes"] = peak_working_set_bytes
    environment["peak_working_set_mb"] = (
        round(peak_working_set_bytes / (1024 * 1024), 2)
        if peak_working_set_bytes is not None
        else None
    )
    environment["peak_working_set_measurement"] = (
        "Windows GetProcessMemoryInfo PeakWorkingSetSize"
        if peak_working_set_bytes is not None
        else None
    )
    command_args = list(argv) if argv is not None else sys.argv[1:]
    command = " ".join(
        [
            sys.executable,
            "-m",
            "evaluation.evidence_conflict.benchmark",
            *command_args,
        ]
    )
    report = build_report(
        dataset,
        evaluation,
        config,
        environment=environment,
        command=command,
    )
    json_path, markdown_path = write_reports(
        report,
        args.output_dir,
        prefix=args.output_prefix,
    )

    summary = {
        "dataset": str(dataset.path),
        "total_examples": evaluation.metrics["total_examples"],
        "evaluated_examples": evaluation.metrics["evaluated_examples"],
        "inference_failures": evaluation.metrics["inference_failures"],
        "accuracy": evaluation.metrics["accuracy"],
        "macro_f1": evaluation.metrics["macro_f1"],
        "json_report": str(json_path),
        "markdown_report": str(markdown_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
