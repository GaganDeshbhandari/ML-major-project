"""Compare two NLI stance benchmark reports.

The comparison uses the JSON reports produced by
``evaluation.evidence_conflict.benchmark``. It does not load a model or
recompute predictions; it only summarizes and compares measured results.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_OUTPUT_PREFIX = "nli_model_comparison"


class ComparisonError(ValueError):
    """Raised when benchmark reports cannot be compared safely."""


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComparisonError(f"{field} must be a JSON object.")
    return value


def load_report(path: str | Path) -> dict[str, Any]:
    """Load one benchmark JSON report."""

    report_path = Path(path)
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ComparisonError(f"Could not read report {report_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ComparisonError(
            f"Report {report_path} is not valid JSON: {exc}"
        ) from exc

    report = dict(_require_mapping(payload, str(report_path)))
    for field in ("dataset", "model", "metrics", "runtime", "environment"):
        _require_mapping(report.get(field), f"{report_path}.{field}")
    for field in ("misclassified_examples", "inference_failures"):
        if not isinstance(report.get(field), list):
            raise ComparisonError(f"{report_path}.{field} must be a list.")
    return report


def _metric(report: Mapping[str, Any], key: str) -> Any:
    return report.get("metrics", {}).get(key)


def _runtime_metric(report: Mapping[str, Any], key: str) -> Any:
    return report.get("runtime", {}).get(key)


def _environment_metric(report: Mapping[str, Any], key: str) -> Any:
    return report.get("environment", {}).get(key)


def _per_class_map(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    per_class = _metric(report, "per_class")
    if not isinstance(per_class, list):
        raise ComparisonError("metrics.per_class must be a list.")
    result: dict[str, Mapping[str, Any]] = {}
    for item in per_class:
        mapping = _require_mapping(item, "metrics.per_class item")
        label = mapping.get("label")
        if not isinstance(label, str):
            raise ComparisonError("metrics.per_class item has no string label.")
        result[label] = mapping
    return result


def _confusion_count(
    confusions: Sequence[Mapping[str, Any]] | None,
    gold: str,
    predicted: str,
) -> int:
    for item in confusions or []:
        if (
            item.get("gold_label") == gold
            and item.get("predicted_label") == predicted
        ):
            count = item.get("count")
            if isinstance(count, int) and not isinstance(count, bool):
                return count
    return 0


def _error_records(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return non-failure misclassified records keyed by stable example ID."""

    examples = report.get("examples")
    if not isinstance(examples, list):
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for record in examples:
        if not isinstance(record, Mapping):
            continue
        if record.get("error_type") is not None:
            continue
        if record.get("correct") is True:
            continue
        record_id = record.get("id")
        if record_id is None:
            continue
        result[str(record_id)] = record
    return result


def _high_confidence_error_count(
    records: Mapping[str, Mapping[str, Any]],
    threshold: float = 0.9,
) -> int:
    return sum(
        1
        for record in records.values()
        if isinstance(record.get("confidence"), (int, float))
        and not isinstance(record.get("confidence"), bool)
        and float(record["confidence"]) >= threshold
    )


def _delta(baseline: Any, alternative: Any) -> float | int | None:
    if (
        isinstance(baseline, (int, float))
        and not isinstance(baseline, bool)
        and isinstance(alternative, (int, float))
        and not isinstance(alternative, bool)
    ):
        return alternative - baseline
    return None


def _headline_row(
    label: str,
    baseline: Any,
    alternative: Any,
    *,
    unit: str | None = None,
) -> dict[str, Any]:
    return {
        "metric": label,
        "unit": unit,
        "baseline": baseline,
        "alternative": alternative,
        "delta_alternative_minus_baseline": _delta(baseline, alternative),
    }


def _validate_same_dataset(
    baseline: Mapping[str, Any],
    alternative: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    baseline_dataset = _require_mapping(baseline["dataset"], "baseline.dataset")
    alternative_dataset = _require_mapping(
        alternative["dataset"],
        "alternative.dataset",
    )
    baseline_sha = baseline_dataset.get("sha256")
    alternative_sha = alternative_dataset.get("sha256")
    if not baseline_sha or not alternative_sha:
        raise ComparisonError("Both reports must record a dataset SHA-256.")
    if baseline_sha != alternative_sha:
        raise ComparisonError(
            "Reports were produced on different datasets: "
            f"{baseline_sha!r} != {alternative_sha!r}."
        )
    if baseline_dataset.get("total_examples") != alternative_dataset.get(
        "total_examples"
    ):
        raise ComparisonError("Reports disagree on the number of examples.")
    return baseline_dataset, alternative_dataset


def build_comparison(
    baseline: Mapping[str, Any],
    alternative: Mapping[str, Any],
    *,
    baseline_label: str = "Existing xsmall",
    alternative_label: str = "Alternative base",
    generated_at: str | None = None,
    baseline_note: str | None = None,
    extra_notes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a machine-readable comparison from two benchmark reports."""

    baseline_dataset, alternative_dataset = _validate_same_dataset(
        baseline,
        alternative,
    )
    baseline_per_class = _per_class_map(baseline)
    alternative_per_class = _per_class_map(alternative)
    baseline_confusions = baseline["metrics"].get("confusions")
    alternative_confusions = alternative["metrics"].get("confusions")
    baseline_errors = _error_records(baseline)
    alternative_errors = _error_records(alternative)
    common_error_ids = sorted(set(baseline_errors) & set(alternative_errors))
    fixed_error_ids = sorted(set(baseline_errors) - set(alternative_errors))
    new_error_ids = sorted(set(alternative_errors) - set(baseline_errors))

    def class_metric(
        per_class: Mapping[str, Mapping[str, Any]],
        label: str,
        key: str,
    ) -> Any:
        return per_class.get(label, {}).get(key)

    baseline_f1 = {
        label: class_metric(baseline_per_class, label, "f1")
        for label in ("supports", "refutes", "not_enough_info")
    }
    alternative_f1 = {
        label: class_metric(alternative_per_class, label, "f1")
        for label in ("supports", "refutes", "not_enough_info")
    }
    baseline_precision = {
        label: class_metric(baseline_per_class, label, "precision")
        for label in ("supports", "refutes", "not_enough_info")
    }
    alternative_precision = {
        label: class_metric(alternative_per_class, label, "precision")
        for label in ("supports", "refutes", "not_enough_info")
    }
    baseline_confusion_counts = {
        "nei_to_supports": _confusion_count(
            baseline_confusions,
            "not_enough_info",
            "supports",
        ),
        "nei_to_refutes": _confusion_count(
            baseline_confusions,
            "not_enough_info",
            "refutes",
        ),
        "refutes_to_supports": _confusion_count(
            baseline_confusions,
            "refutes",
            "supports",
        ),
        "supports_to_nei": _confusion_count(
            baseline_confusions,
            "supports",
            "not_enough_info",
        ),
    }
    alternative_confusion_counts = {
        "nei_to_supports": _confusion_count(
            alternative_confusions,
            "not_enough_info",
            "supports",
        ),
        "nei_to_refutes": _confusion_count(
            alternative_confusions,
            "not_enough_info",
            "refutes",
        ),
        "refutes_to_supports": _confusion_count(
            alternative_confusions,
            "refutes",
            "supports",
        ),
        "supports_to_nei": _confusion_count(
            alternative_confusions,
            "supports",
            "not_enough_info",
        ),
    }

    headline = [
        _headline_row(
            "Accuracy",
            _metric(baseline, "accuracy"),
            _metric(alternative, "accuracy"),
        ),
        _headline_row(
            "Macro-F1",
            _metric(baseline, "macro_f1"),
            _metric(alternative, "macro_f1"),
        ),
        _headline_row(
            "SUPPORTS recall",
            class_metric(baseline_per_class, "supports", "recall"),
            class_metric(alternative_per_class, "supports", "recall"),
        ),
        _headline_row(
            "REFUTES recall",
            class_metric(baseline_per_class, "refutes", "recall"),
            class_metric(alternative_per_class, "refutes", "recall"),
        ),
        _headline_row(
            "NOT_ENOUGH_INFO recall",
            class_metric(
                baseline_per_class,
                "not_enough_info",
                "recall",
            ),
            class_metric(
                alternative_per_class,
                "not_enough_info",
                "recall",
            ),
        ),
        _headline_row(
            "Inference failures",
            _metric(baseline, "inference_failures"),
            _metric(alternative, "inference_failures"),
        ),
        _headline_row(
            "Coverage",
            _metric(baseline, "coverage"),
            _metric(alternative, "coverage"),
        ),
        _headline_row(
            "Evaluated examples",
            _metric(baseline, "evaluated_examples"),
            _metric(alternative, "evaluated_examples"),
        ),
        _headline_row(
            "Average warm inference seconds/example",
            _runtime_metric(baseline, "average_warm_inference_seconds"),
            _runtime_metric(alternative, "average_warm_inference_seconds"),
            unit="seconds",
        ),
        _headline_row(
            "Initial loading seconds",
            _runtime_metric(baseline, "initial_call_seconds"),
            _runtime_metric(alternative, "initial_call_seconds"),
            unit="seconds",
        ),
        _headline_row(
            "Total attempt seconds",
            _runtime_metric(baseline, "total_attempt_seconds"),
            _runtime_metric(alternative, "total_attempt_seconds"),
            unit="seconds",
        ),
        _headline_row(
            "Peak working set MB",
            _environment_metric(baseline, "peak_working_set_mb"),
            _environment_metric(alternative, "peak_working_set_mb"),
            unit="MB",
        ),
    ]

    baseline_accuracy = _metric(baseline, "accuracy")
    alternative_accuracy = _metric(alternative, "accuracy")
    baseline_macro_f1 = _metric(baseline, "macro_f1")
    alternative_macro_f1 = _metric(alternative, "macro_f1")
    baseline_nei_recall = class_metric(
        baseline_per_class,
        "not_enough_info",
        "recall",
    )
    alternative_nei_recall = class_metric(
        alternative_per_class,
        "not_enough_info",
        "recall",
    )

    if (
        isinstance(alternative_accuracy, (int, float))
        and isinstance(baseline_accuracy, (int, float))
        and isinstance(alternative_macro_f1, (int, float))
        and isinstance(baseline_macro_f1, (int, float))
    ):
        if (
            alternative_accuracy > baseline_accuracy
            and alternative_macro_f1 > baseline_macro_f1
        ):
            verdict = "alternative"
        elif (
            alternative_accuracy < baseline_accuracy
            and alternative_macro_f1 < baseline_macro_f1
        ):
            verdict = "baseline"
        else:
            verdict = "mixed"
    else:
        verdict = "not_determined"

    warm_ratio = None
    baseline_warm = _runtime_metric(
        baseline,
        "average_warm_inference_seconds",
    )
    alternative_warm = _runtime_metric(
        alternative,
        "average_warm_inference_seconds",
    )
    if (
        isinstance(baseline_warm, (int, float))
        and baseline_warm > 0
        and isinstance(alternative_warm, (int, float))
    ):
        warm_ratio = alternative_warm / baseline_warm

    total_ratio = None
    baseline_total = _runtime_metric(baseline, "total_attempt_seconds")
    alternative_total = _runtime_metric(alternative, "total_attempt_seconds")
    if (
        isinstance(baseline_total, (int, float))
        and baseline_total > 0
        and isinstance(alternative_total, (int, float))
    ):
        total_ratio = alternative_total / baseline_total

    accuracy_delta = _delta(baseline_accuracy, alternative_accuracy)
    macro_f1_delta = _delta(baseline_macro_f1, alternative_macro_f1)
    nei_recall_delta = _delta(baseline_nei_recall, alternative_nei_recall)
    nei_f1_delta = _delta(
        baseline_f1["not_enough_info"],
        alternative_f1["not_enough_info"],
    )
    supports_f1_delta = _delta(
        baseline_f1["supports"],
        alternative_f1["supports"],
    )
    refutes_f1_delta = _delta(
        baseline_f1["refutes"],
        alternative_f1["refutes"],
    )

    if nei_recall_delta is not None and nei_recall_delta > 0:
        if nei_f1_delta is not None and nei_f1_delta > 0:
            if (
                accuracy_delta is not None
                and nei_recall_delta >= accuracy_delta
            ):
                improvement_concentrated_in_nei_recall = "yes"
            else:
                improvement_concentrated_in_nei_recall = "partly"
        else:
            improvement_concentrated_in_nei_recall = "no"
    else:
        improvement_concentrated_in_nei_recall = "no"

    if verdict == "alternative":
        if (
            improvement_concentrated_in_nei_recall == "no"
            and warm_ratio is not None
            and warm_ratio > 5
        ):
            assessment = (
                "Marginal headline improvement; the gain is not concentrated "
                "in NOT_ENOUGH_INFO recall, and warm inference is more than "
                "5x slower. On this evidence, the alternative does not "
                "justify replacing the production default."
            )
        elif (
            improvement_concentrated_in_nei_recall in {"yes", "partly"}
            and warm_ratio is not None
            and warm_ratio <= 2
        ):
            assessment = (
                "The alternative improves the target behavior with a modest "
                "warm-runtime cost; it is a candidate for further evaluation."
            )
        else:
            assessment = (
                "The alternative is higher on headline metrics; review the "
                "per-class trade-offs and runtime before deciding."
            )
    elif verdict == "baseline":
        assessment = (
            "The baseline is higher on both headline metrics on this dataset."
        )
    elif verdict == "mixed":
        assessment = (
            "The headline metrics disagree; the choice depends on which "
            "class errors and runtime cost matter most."
        )
    else:
        assessment = "The available reports do not support a verdict."

    total_examples = baseline_dataset.get("total_examples")
    baseline_threshold = baseline["model"].get("confidence_threshold")
    alternative_threshold = alternative["model"].get("confidence_threshold")
    if baseline_threshold is None and alternative_threshold is None:
        threshold_note = "No confidence threshold was configured for either model."
    else:
        threshold_note = (
            "Configured confidence thresholds: baseline="
            f"{baseline_threshold!r}, alternative={alternative_threshold!r}."
        )

    baseline_dtype = baseline["model"].get("model_dtype")
    alternative_dtype = alternative["model"].get("model_dtype")
    if baseline_dtype != alternative_dtype:
        dtype_note = (
            f"Model dtypes differ: baseline={baseline_dtype}, "
            f"alternative={alternative_dtype}. The existing classifier loaded "
            "each checkpoint's configured dtype; CPU runtime comparisons may "
            "be affected by dtype-specific kernels."
        )
    else:
        dtype_note = f"Both models loaded with dtype {baseline_dtype}."

    measurement_notes = [
        "All classification metrics and timings are read from the benchmark "
        "JSON reports; they are not recomputed here.",
        "Peak memory is the process peak working set measured with Windows "
        "GetProcessMemoryInfo. It includes the Python process, imported "
        "libraries, tokenizer, model, and inference workspace; it is not "
        "isolated model-weight memory.",
        f"The dataset is the frozen {total_examples}-example diagnostic "
        "set. Results are not a representative benchmark or proof of "
        "general reliability.",
        "With 90 examples, one example changes accuracy by about 1.11 "
        "percentage points; small differences are not statistically robust.",
        threshold_note,
        dtype_note,
    ]
    if baseline_note:
        measurement_notes.append(baseline_note)
    if extra_notes:
        measurement_notes.extend(extra_notes)

    return {
        "comparison_schema_version": "1.0",
        "generated_at_utc": generated_at
        or datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "sha256": baseline_dataset["sha256"],
            "total_examples": baseline_dataset.get("total_examples"),
            "class_counts": baseline_dataset.get("class_counts"),
            "baseline_report_dataset_path": baseline_dataset.get("path"),
            "alternative_report_dataset_path": alternative_dataset.get("path"),
        },
        "models": {
            "baseline": {
                "label": baseline_label,
                "model": baseline["model"],
            },
            "alternative": {
                "label": alternative_label,
                "model": alternative["model"],
            },
        },
        "headline_metrics": headline,
        "per_class": {
            "baseline": baseline["metrics"].get("per_class"),
            "alternative": alternative["metrics"].get("per_class"),
        },
        "confusion": {
            "baseline": {
                "labels": baseline["metrics"].get("confusion_matrix_labels"),
                "matrix": baseline["metrics"].get("confusion_matrix"),
                "confusions": baseline["metrics"].get("confusions"),
            },
            "alternative": {
                "labels": alternative["metrics"].get(
                    "confusion_matrix_labels"
                ),
                "matrix": alternative["metrics"].get("confusion_matrix"),
                "confusions": alternative["metrics"].get("confusions"),
            },
        },
        "misclassified_examples": {
            "baseline": baseline["misclassified_examples"],
            "alternative": alternative["misclassified_examples"],
        },
        "inference_failures": {
            "baseline": baseline["inference_failures"],
            "alternative": alternative["inference_failures"],
        },
        "run_counts": {
            "baseline": {
                "total_examples": _metric(baseline, "total_examples"),
                "evaluated_examples": _metric(baseline, "evaluated_examples"),
                "inference_failures": _metric(baseline, "inference_failures"),
                "coverage": _metric(baseline, "coverage"),
            },
            "alternative": {
                "total_examples": _metric(alternative, "total_examples"),
                "evaluated_examples": _metric(
                    alternative,
                    "evaluated_examples",
                ),
                "inference_failures": _metric(
                    alternative,
                    "inference_failures",
                ),
                "coverage": _metric(alternative, "coverage"),
            },
        },
        "per_class_f1": {
            "baseline": baseline_f1,
            "alternative": alternative_f1,
            "delta_alternative_minus_baseline": {
                "supports": supports_f1_delta,
                "refutes": refutes_f1_delta,
                "not_enough_info": nei_f1_delta,
            },
        },
        "per_class_precision": {
            "baseline": baseline_precision,
            "alternative": alternative_precision,
        },
        "confusion_counts": {
            "baseline": baseline_confusion_counts,
            "alternative": alternative_confusion_counts,
        },
        "error_transitions": {
            "baseline_misclassified_count": len(baseline_errors),
            "alternative_misclassified_count": len(alternative_errors),
            "common_error_ids": common_error_ids,
            "fixed_by_alternative_ids": fixed_error_ids,
            "new_error_ids_in_alternative": new_error_ids,
            "baseline_high_confidence_error_count": (
                _high_confidence_error_count(baseline_errors)
            ),
            "alternative_high_confidence_error_count": (
                _high_confidence_error_count(alternative_errors)
            ),
        },
        "summary": {
            "verdict": verdict,
            "assessment": assessment,
            "improvement_concentrated_in_not_enough_info_recall": (
                improvement_concentrated_in_nei_recall
            ),
            "accuracy_delta": accuracy_delta,
            "macro_f1_delta": macro_f1_delta,
            "not_enough_info_recall_delta": nei_recall_delta,
            "supports_f1_delta": supports_f1_delta,
            "refutes_f1_delta": refutes_f1_delta,
            "not_enough_info_f1_delta": nei_f1_delta,
            "warm_runtime_ratio_alternative_over_baseline": warm_ratio,
            "total_runtime_ratio_alternative_over_baseline": total_ratio,
            "peak_memory_delta_mb": _delta(
                _environment_metric(baseline, "peak_working_set_mb"),
                _environment_metric(alternative, "peak_working_set_mb"),
            ),
        },
        "measurement_notes": measurement_notes,
        "provenance": {
            "baseline_report": str(baseline.get("_report_path", "")),
            "alternative_report": str(alternative.get("_report_path", "")),
            "baseline_note": baseline_note,
        },
    }


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _format_headline_value(item: Mapping[str, Any], value: Any) -> str:
    metric = item.get("metric")
    if metric in {"Inference failures", "Evaluated examples"}:
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return _format_number(value, 0)
    if item.get("unit") == "MB":
        return _format_number(value, 1)
    if item.get("unit") == "seconds":
        return _format_number(value, 3)
    return _format_number(value, 4)


def _md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(_md_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_md_cell(cell) for cell in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _model_table_row(model: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        model.get("name"),
        model.get("configured_revision"),
        model.get("device_requested"),
        model.get("resolved_device"),
        model.get("model_dtype"),
        model.get("sequence_length"),
        model.get("batch_size"),
        model.get("confidence_threshold"),
    )


def _render_per_class(
    label: str,
    per_class: Sequence[Mapping[str, Any]],
) -> list[str]:
    lines = [
        f"### {label}",
        "",
        _md_table(
            ("Label", "Precision", "Recall", "F1", "Support"),
            [
                (
                    item.get("label"),
                    _format_number(item.get("precision")),
                    _format_number(item.get("recall")),
                    _format_number(item.get("f1")),
                    item.get("support"),
                )
                for item in per_class
            ],
        ),
        "",
    ]
    return lines


def _render_confusion(
    label: str,
    confusion: Mapping[str, Any],
) -> list[str]:
    labels = confusion.get("labels") or []
    matrix = confusion.get("matrix") or []
    lines = [
        f"### {label}",
        "",
        "Rows are gold labels; columns are predicted labels.",
        "",
        _md_table(
            ("Gold \\ Predicted", *labels),
            [
                (gold, *row)
                for gold, row in zip(labels, matrix)
            ],
        ),
        "",
    ]
    confusions = confusion.get("confusions") or []
    if confusions:
        lines.append(
            _md_table(
                ("Gold", "Predicted", "Count"),
                [
                    (
                        item.get("gold_label"),
                        item.get("predicted_label"),
                        item.get("count"),
                    )
                    for item in confusions
                ],
            )
        )
    else:
        lines.append("No off-diagonal confusion entries.")
    lines.append("")
    return lines


def _selected_errors(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        gold = record.get("gold_label")
        predicted = record.get("predicted_label")
        confidence = record.get("confidence")
        interesting = (
            (gold == "not_enough_info" and predicted == "supports")
            or (gold == "not_enough_info" and predicted == "refutes")
            or (gold == "refutes" and predicted == "supports")
            or (
                isinstance(confidence, (int, float))
                and confidence >= 0.9
            )
        )
        if not interesting:
            continue
        record_id = str(record.get("id"))
        if record_id in seen:
            continue
        seen.add(record_id)
        selected.append(record)
    return selected


def _render_errors(
    label: str,
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    lines = [f"### {label}", ""]
    if not records:
        lines.extend(["No misclassified examples.", ""])
        return lines

    lines.append(
        _md_table(
            ("ID", "Gold", "Predicted", "Confidence", "Category"),
            [
                (
                    item.get("id"),
                    item.get("gold_label"),
                    item.get("predicted_label"),
                    _format_number(item.get("confidence")),
                    item.get("category"),
                )
                for item in records
            ],
        )
    )
    lines.append("")
    selected = _selected_errors(records)
    if selected:
        lines.extend(
            [
                "Selected high-priority or high-confidence errors:",
                "",
            ]
        )
        for item in selected:
            lines.extend(
                [
                    f"- `{item.get('id')}` gold=`{item.get('gold_label')}` "
                    f"predicted=`{item.get('predicted_label')}` "
                    f"confidence={_format_number(item.get('confidence'))}",
                    f"  - Claim: {item.get('claim')}",
                    f"  - Evidence: {item.get('evidence')}",
                ]
            )
        lines.append("")
    return lines


_CONFUSION_DISPLAY = {
    "nei_to_supports": "NOT_ENOUGH_INFO -> SUPPORTS",
    "nei_to_refutes": "NOT_ENOUGH_INFO -> REFUTES",
    "refutes_to_supports": "REFUTES -> SUPPORTS",
    "supports_to_nei": "SUPPORTS -> NOT_ENOUGH_INFO",
}


def _render_confusion_counts(
    comparison: Mapping[str, Any],
    baseline_label: str,
    alternative_label: str,
) -> list[str]:
    counts = comparison["confusion_counts"]
    return [
        "## Confusion count comparison",
        "",
        _md_table(
            ("Confusion", baseline_label, alternative_label),
            [
                (
                    display,
                    counts["baseline"].get(key, 0),
                    counts["alternative"].get(key, 0),
                )
                for key, display in _CONFUSION_DISPLAY.items()
            ],
        ),
        "",
    ]


def _render_error_transitions(
    comparison: Mapping[str, Any],
) -> list[str]:
    transitions = comparison["error_transitions"]
    lines = [
        "## Error transitions",
        "",
        _md_table(
            ("Item", "Value"),
            (
                (
                    "Baseline misclassified",
                    transitions.get("baseline_misclassified_count"),
                ),
                (
                    "Alternative misclassified",
                    transitions.get("alternative_misclassified_count"),
                ),
                (
                    "Baseline high-confidence errors (>= 0.9)",
                    transitions.get("baseline_high_confidence_error_count"),
                ),
                (
                    "Alternative high-confidence errors (>= 0.9)",
                    transitions.get("alternative_high_confidence_error_count"),
                ),
            ),
        ),
        "",
    ]
    for key, label in (
        ("common_error_ids", "Common errors"),
        ("fixed_by_alternative_ids", "Fixed by alternative"),
        ("new_error_ids_in_alternative", "New in alternative"),
    ):
        ids = transitions.get(key) or []
        rendered = ", ".join(f"`{item}`" for item in ids) if ids else "none"
        lines.append(f"- {label}: {rendered}")
    lines.append("")
    return lines


def render_comparison_markdown(comparison: Mapping[str, Any]) -> str:
    """Render a readable model-comparison report."""

    dataset = comparison["dataset"]
    models = comparison["models"]
    baseline_label = models["baseline"]["label"]
    alternative_label = models["alternative"]["label"]
    headline = comparison["headline_metrics"]
    summary = comparison["summary"]

    lines: list[str] = [
        "# NLI Stance Model Comparison",
        "",
        f"Generated: {comparison['generated_at_utc']}",
        "",
        "## Scope",
        "",
        "This comparison uses the frozen diagnostic dataset and the existing "
        "benchmark metric definitions. It compares measured results only; it "
        "does not establish general model quality.",
        "",
        "## Dataset",
        "",
        _md_table(
            ("Property", "Value"),
            (
                ("SHA-256", dataset.get("sha256")),
                ("Total examples", dataset.get("total_examples")),
                ("Class counts", json.dumps(dataset.get("class_counts"))),
            ),
        ),
        "",
        "## Models",
        "",
        _md_table(
            (
                "Label",
                "Model",
                "Configured revision",
                "Device requested",
                "Resolved device",
                "Dtype",
                "Sequence length",
                "Configured batch size",
                "Confidence threshold",
            ),
            (
                (baseline_label, *_model_table_row(models["baseline"]["model"])),
                (
                    alternative_label,
                    *_model_table_row(models["alternative"]["model"]),
                ),
            ),
        ),
        "",
        "## Run counts",
        "",
        _md_table(
            (
                "Model",
                "Total examples",
                "Evaluated",
                "Inference failures",
                "Coverage",
            ),
            [
                (
                    label,
                    counts.get("total_examples"),
                    counts.get("evaluated_examples"),
                    counts.get("inference_failures"),
                    _format_number(counts.get("coverage")),
                )
                for label, counts in (
                    (baseline_label, comparison["run_counts"]["baseline"]),
                    (
                        alternative_label,
                        comparison["run_counts"]["alternative"],
                    ),
                )
            ],
        ),
        "",
        "## Headline comparison",
        "",
        _md_table(
            ("Metric", baseline_label, alternative_label, "Delta (alt - baseline)"),
            [
                (
                    item["metric"],
                    _format_headline_value(item, item["baseline"]),
                    _format_headline_value(item, item["alternative"]),
                    _format_headline_value(
                        item,
                        item["delta_alternative_minus_baseline"],
                    ),
                )
                for item in headline
            ],
        ),
        "",
        "## Per-class metrics",
        "",
    ]

    lines.extend(
        _render_per_class(
            baseline_label,
            comparison["per_class"]["baseline"],
        )
    )
    lines.extend(
        _render_per_class(
            alternative_label,
            comparison["per_class"]["alternative"],
        )
    )

    lines.extend(
        [
            "## Confusion matrices",
            "",
        ]
    )
    lines.extend(
        _render_confusion(
            baseline_label,
            comparison["confusion"]["baseline"],
        )
    )
    lines.extend(
        _render_confusion(
            alternative_label,
            comparison["confusion"]["alternative"],
        )
    )
    lines.extend(
        _render_confusion_counts(
            comparison,
            baseline_label,
            alternative_label,
        )
    )
    lines.extend(
        _render_error_transitions(comparison)
    )

    lines.extend(
        [
            "## Misclassified examples",
            "",
            "The tables preserve example IDs. The selected bullets emphasize "
            "NOT_ENOUGH_INFO-to-stance errors, REFUTES-to-SUPPORTS errors, "
            "and incorrect predictions with confidence >= 0.9.",
            "",
        ]
    )
    lines.extend(
        _render_errors(
            baseline_label,
            comparison["misclassified_examples"]["baseline"],
        )
    )
    lines.extend(
        _render_errors(
            alternative_label,
            comparison["misclassified_examples"]["alternative"],
        )
    )

    baseline_counts = comparison["confusion_counts"]["baseline"]
    alternative_counts = comparison["confusion_counts"]["alternative"]
    baseline_precision = comparison["per_class_precision"]["baseline"]
    alternative_precision = comparison["per_class_precision"]["alternative"]
    accuracy_delta = summary.get("accuracy_delta")
    total_examples = dataset.get("total_examples")
    accuracy_delta_examples = None
    if (
        isinstance(accuracy_delta, (int, float))
        and isinstance(total_examples, int)
        and total_examples > 0
    ):
        accuracy_delta_examples = round(accuracy_delta * total_examples)
    if accuracy_delta_examples is not None:
        example_word = "example" if abs(accuracy_delta_examples) == 1 else "examples"
        accuracy_delta_examples_text = (
            f"{accuracy_delta_examples:+d} {example_word}"
        )
    else:
        accuracy_delta_examples_text = "n/a"
    warm_ratio_text = _format_number(
        summary.get("warm_runtime_ratio_alternative_over_baseline")
    )
    total_ratio_text = _format_number(
        summary.get("total_runtime_ratio_alternative_over_baseline")
    )
    baseline_per_class = {
        item["label"]: item
        for item in comparison["per_class"]["baseline"]
    }
    alternative_per_class = {
        item["label"]: item
        for item in comparison["per_class"]["alternative"]
    }
    baseline_nei = baseline_per_class.get("not_enough_info", {})
    alternative_nei = alternative_per_class.get("not_enough_info", {})
    transitions = comparison["error_transitions"]

    lines.extend(
        [
            "## Reading the comparison",
            "",
            f"- Rule-based headline verdict: `{summary.get('verdict')}`.",
            f"- Assessment: {summary.get('assessment')}",
            "- Headline: the alternative is higher on accuracy and macro-F1. "
            "For the stated NOT_ENOUGH_INFO concern, the baseline has higher "
            "NOT_ENOUGH_INFO precision "
            f"(`{_format_number(baseline_nei.get('precision'))}` vs "
            f"`{_format_number(alternative_nei.get('precision'))}`) "
            "while the alternative has slightly higher recall "
            f"(`{_format_number(baseline_nei.get('recall'))}` vs "
            f"`{_format_number(alternative_nei.get('recall'))}`); "
            "the alternative is not a clear win on that concern.",
            f"- Accuracy delta (alternative - baseline): "
            f"`{_format_number(accuracy_delta)}` "
            f"({accuracy_delta_examples_text}).",
            f"- Macro-F1 delta: "
            f"`{_format_number(summary.get('macro_f1_delta'))}`.",
            f"- NOT_ENOUGH_INFO recall delta: "
            f"`{_format_number(summary.get('not_enough_info_recall_delta'))}`; "
            "NOT_ENOUGH_INFO F1 delta: "
            f"`{_format_number(summary.get('not_enough_info_f1_delta'))}`; "
            "precision "
            f"`{_format_number(baseline_precision.get('not_enough_info'))}` -> "
            f"`{_format_number(alternative_precision.get('not_enough_info'))}`.",
            f"- Improvement concentrated in NOT_ENOUGH_INFO recall: "
            f"`{summary.get('improvement_concentrated_in_not_enough_info_recall')}`.",
            f"- Per-class F1 deltas: SUPPORTS "
            f"`{_format_number(summary.get('supports_f1_delta'))}`, REFUTES "
            f"`{_format_number(summary.get('refutes_f1_delta'))}`, "
            "NOT_ENOUGH_INFO "
            f"`{_format_number(summary.get('not_enough_info_f1_delta'))}`.",
            f"- Confusion counts: NOT_ENOUGH_INFO -> SUPPORTS "
            f"`{baseline_counts.get('nei_to_supports')}` -> "
            f"`{alternative_counts.get('nei_to_supports')}`; "
            "NOT_ENOUGH_INFO -> REFUTES "
            f"`{baseline_counts.get('nei_to_refutes')}` -> "
            f"`{alternative_counts.get('nei_to_refutes')}`; "
            "REFUTES -> SUPPORTS "
            f"`{baseline_counts.get('refutes_to_supports')}` -> "
            f"`{alternative_counts.get('refutes_to_supports')}`; "
            "SUPPORTS -> NOT_ENOUGH_INFO "
            f"`{baseline_counts.get('supports_to_nei')}` -> "
            f"`{alternative_counts.get('supports_to_nei')}`.",
            f"- Warm runtime ratio (alternative / baseline): "
            f"`{warm_ratio_text}`; total runtime ratio: "
            f"`{total_ratio_text}`.",
            f"- Peak working-set delta MB: "
            f"`{_format_number(summary.get('peak_memory_delta_mb'), 1)}`.",
            f"- Misclassified examples: baseline "
            f"`{transitions.get('baseline_misclassified_count')}`, alternative "
            f"`{transitions.get('alternative_misclassified_count')}`; "
            "high-confidence errors (>= 0.9): baseline "
            f"`{transitions.get('baseline_high_confidence_error_count')}`, "
            "alternative "
            f"`{transitions.get('alternative_high_confidence_error_count')}`.",
            "",
            "The verdict is a simple rule: `alternative` means both accuracy "
            "and macro-F1 are higher; `baseline` means both are lower; "
            "`mixed` means the two headline metrics disagree. The assessment "
            "is a rule-based interpretation, not a statistical test.",
            "",
            "## Measurement notes and caveats",
            "",
        ]
    )
    lines.extend(
        f"- {note}" for note in comparison["measurement_notes"]
    )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            _md_table(
                ("Report", "Path"),
                (
                    (baseline_label, comparison["provenance"]["baseline_report"]),
                    (
                        alternative_label,
                        comparison["provenance"]["alternative_report"],
                    ),
                ),
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_comparison(
    comparison: Mapping[str, Any],
    output_dir: str | Path,
    *,
    prefix: str = DEFAULT_OUTPUT_PREFIX,
) -> tuple[Path, Path]:
    """Write comparison JSON and Markdown reports."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / f"{prefix}.json"
    markdown_path = destination / f"{prefix}.md"
    json_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_comparison_markdown(comparison),
        encoding="utf-8",
    )
    return json_path, markdown_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two NLI stance benchmark JSON reports."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--alternative", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
    )
    parser.add_argument(
        "--baseline-label",
        default="Existing xsmall",
    )
    parser.add_argument(
        "--alternative-label",
        default="Alternative base",
    )
    parser.add_argument(
        "--baseline-note",
        default=None,
        help="Optional provenance note included in the comparison report.",
    )
    parser.add_argument(
        "--measurement-note",
        action="append",
        default=None,
        help="Optional extra measurement note; may be repeated.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        baseline = load_report(args.baseline)
        alternative = load_report(args.alternative)
        baseline["_report_path"] = str(args.baseline)
        alternative["_report_path"] = str(args.alternative)
        comparison = build_comparison(
            baseline,
            alternative,
            baseline_label=args.baseline_label,
            alternative_label=args.alternative_label,
            baseline_note=args.baseline_note,
            extra_notes=args.measurement_note,
        )
    except ComparisonError as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        return 2

    json_path, markdown_path = write_comparison(
        comparison,
        args.output_dir,
        prefix=args.output_prefix,
    )
    print(
        json.dumps(
            {
                "baseline": str(args.baseline),
                "alternative": str(args.alternative),
                "verdict": comparison["summary"]["verdict"],
                "json_report": str(json_path),
                "markdown_report": str(markdown_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
