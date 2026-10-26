"""Validation and provenance for manually labeled claim-ranking datasets.

The dataset contract is intentionally independent of the production pipeline.
A dataset records human priority labels, optional ECS values, and the baseline
ranks produced by the existing O1/O2/O3 rankings. It does not generate labels
from ECS or from any model output.

No claim-ranking dataset is present in the repository. The default path is the
expected location for a future, manually annotated dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .metrics import (
    DEFAULT_K_VALUES,
    DEFAULT_RELEVANT_MIN_GRADE,
    MAX_RELEVANCE_GRADE,
    evaluate_ranking,
    validate_k_values,
)

#: The three existing objective-specific rankings. There is no combined score.
OBJECTIVES: tuple[str, ...] = ("O1", "O2", "O3")

#: Version of the documented dataset schema.
DATASET_SCHEMA_VERSION = "1.0"

#: Identifiers for the planned manually labeled dataset.
DEFAULT_DATASET_ID = "claim-ranking-eval"
DEFAULT_DATASET_VERSION = "1"
DEFAULT_LABELING_GUIDE_VERSION = "1"

#: Expected location for a future labeled dataset. This file is not present.
DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent / "data" / "claim_ranking_dataset_v1.json"
)


class DatasetValidationError(ValueError):
    """Raised when a claim-ranking dataset is malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Optional evidence retained to reproduce a labeled example.

    Full text may be omitted when licensing or availability prevents
    redistribution; use ``reproduction_notes`` for a citation instead.
    """

    content: str
    title: str = ""
    url: str = ""
    retrieved_at: str | None = None
    license: str | None = None


@dataclass(frozen=True, slots=True)
class Exclusion:
    """A claim considered for the dataset but not evaluated, with a reason."""

    claim_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class ClaimRankingExample:
    """One manually labeled claim and its recorded baseline ranks."""

    claim_id: str
    claim: str
    relevance_grade: int
    rationale: str
    baseline_ranks: Mapping[str, int | None]
    ecs: float | None
    evidence: tuple[EvidenceRecord, ...] = ()
    reproduction_notes: str | None = None


@dataclass(frozen=True, slots=True)
class ClaimRankingDataset:
    """Validated examples, exclusions, and file provenance."""

    path: Path
    sha256: str
    metadata: Mapping[str, Any]
    examples: tuple[ClaimRankingExample, ...]
    exclusions: tuple[Exclusion, ...] = ()


@dataclass(frozen=True, slots=True)
class BaselineEvaluationSet:
    """The claims and labels eligible for one objective-specific baseline."""

    objective: str
    claim_ids: tuple[str, ...]
    ranked_claim_ids: tuple[str, ...]
    relevance: Mapping[str, int]
    exclusions: tuple[Exclusion, ...]


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{field} must be a non-empty string.")
    return value.strip()


def _optional_nonempty_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, field)


def _optional_string(value: object, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise DatasetValidationError(f"{field} must be a string or null.")
    return value.strip()


def _parse_relevance_grade(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetValidationError(
            f"{field} must be an integer from 0 to {MAX_RELEVANCE_GRADE}."
        )
    if value < 0 or value > MAX_RELEVANCE_GRADE:
        raise DatasetValidationError(
            f"{field} must be from 0 to {MAX_RELEVANCE_GRADE}; got {value}."
        )
    return value


def _parse_ecs(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DatasetValidationError(
            f"{field} must be a finite number in [0, 1] or null."
        )
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DatasetValidationError(
            f"{field} must be a finite number in [0, 1] or null."
        ) from exc
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise DatasetValidationError(
            f"{field} must be a finite number in [0, 1] or null; "
            f"got {value!r}."
        )
    return number


def _parse_baseline_ranks(
    value: object,
    field: str,
) -> Mapping[str, int | None]:
    if not isinstance(value, Mapping):
        raise DatasetValidationError(
            f"{field} must be an object with exactly the keys O1, O2, O3."
        )

    keys = set(value.keys())
    expected = set(OBJECTIVES)
    if keys != expected:
        missing = [objective for objective in OBJECTIVES if objective not in value]
        extra = sorted(str(key) for key in keys - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise DatasetValidationError(
            f"{field} must contain exactly O1, O2, O3 "
            f"({'; '.join(details)})."
        )

    ranks: dict[str, int | None] = {}
    for objective in OBJECTIVES:
        raw_rank = value[objective]
        if raw_rank is None:
            ranks[objective] = None
            continue
        if isinstance(raw_rank, bool) or not isinstance(raw_rank, int):
            raise DatasetValidationError(
                f"{field}.{objective} must be a positive integer or null."
            )
        if raw_rank < 1:
            raise DatasetValidationError(
                f"{field}.{objective} must be >= 1; got {raw_rank}."
            )
        ranks[objective] = raw_rank
    return MappingProxyType(ranks)


def _parse_evidence(
    value: object,
    field: str,
) -> tuple[EvidenceRecord, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise DatasetValidationError(f"{field} must be a list or null.")

    records: list[EvidenceRecord] = []
    for index, raw in enumerate(value):
        item_field = f"{field}[{index}]"
        if not isinstance(raw, Mapping):
            raise DatasetValidationError(f"{item_field} must be an object.")
        if "content" not in raw:
            raise DatasetValidationError(
                f"{item_field} is missing required field 'content'."
            )
        records.append(
            EvidenceRecord(
                content=_nonempty_string(
                    raw["content"],
                    f"{item_field}.content",
                ),
                title=_optional_string(
                    raw.get("title"),
                    f"{item_field}.title",
                ),
                url=_optional_string(
                    raw.get("url"),
                    f"{item_field}.url",
                ),
                retrieved_at=_optional_nonempty_string(
                    raw.get("retrieved_at"),
                    f"{item_field}.retrieved_at",
                ),
                license=_optional_nonempty_string(
                    raw.get("license"),
                    f"{item_field}.license",
                ),
            )
        )
    return tuple(records)


def _parse_example(
    raw: object,
    index: int,
    seen_ids: set[str],
) -> ClaimRankingExample:
    if not isinstance(raw, Mapping):
        raise DatasetValidationError(
            f"examples[{index}] must be an object; "
            f"got {type(raw).__name__}."
        )

    required = (
        "claim_id",
        "claim",
        "relevance_grade",
        "rationale",
        "baseline_ranks",
        "ecs",
    )
    missing = [field for field in required if field not in raw]
    if missing:
        raise DatasetValidationError(
            f"examples[{index}] is missing required field(s): "
            f"{', '.join(missing)}."
        )

    claim_id = _nonempty_string(
        raw["claim_id"],
        f"examples[{index}].claim_id",
    )
    if claim_id in seen_ids:
        raise DatasetValidationError(f"Duplicate claim_id {claim_id!r}.")
    seen_ids.add(claim_id)

    return ClaimRankingExample(
        claim_id=claim_id,
        claim=_nonempty_string(raw["claim"], f"examples[{index}].claim"),
        relevance_grade=_parse_relevance_grade(
            raw["relevance_grade"],
            f"examples[{index}].relevance_grade",
        ),
        rationale=_nonempty_string(
            raw["rationale"],
            f"examples[{index}].rationale",
        ),
        baseline_ranks=_parse_baseline_ranks(
            raw["baseline_ranks"],
            f"examples[{index}].baseline_ranks",
        ),
        ecs=_parse_ecs(raw["ecs"], f"examples[{index}].ecs"),
        evidence=_parse_evidence(
            raw.get("evidence"),
            f"examples[{index}].evidence",
        ),
        reproduction_notes=_optional_nonempty_string(
            raw.get("reproduction_notes"),
            f"examples[{index}].reproduction_notes",
        ),
    )


def _parse_exclusions(
    value: object,
    *,
    example_ids: set[str],
) -> tuple[Exclusion, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise DatasetValidationError("exclusions must be a list or null.")

    seen: set[str] = set()
    exclusions: list[Exclusion] = []
    for index, raw in enumerate(value):
        field = f"exclusions[{index}]"
        if not isinstance(raw, Mapping):
            raise DatasetValidationError(f"{field} must be an object.")
        missing = [
            name
            for name in ("claim_id", "reason")
            if name not in raw
        ]
        if missing:
            raise DatasetValidationError(
                f"{field} is missing required field(s): "
                f"{', '.join(missing)}."
            )
        claim_id = _nonempty_string(
            raw["claim_id"],
            f"{field}.claim_id",
        )
        reason = _nonempty_string(raw["reason"], f"{field}.reason")
        if claim_id in seen:
            raise DatasetValidationError(
                f"Duplicate exclusion claim_id {claim_id!r}."
            )
        if claim_id in example_ids:
            raise DatasetValidationError(
                f"Claim ID {claim_id!r} appears in both examples and exclusions."
            )
        seen.add(claim_id)
        exclusions.append(Exclusion(claim_id=claim_id, reason=reason))
    return tuple(exclusions)


def _parse_payload(
    payload: object,
) -> tuple[
    tuple[ClaimRankingExample, ...],
    tuple[Exclusion, ...],
    dict[str, Any],
]:
    if not isinstance(payload, Mapping):
        raise DatasetValidationError(
            "Dataset must be a JSON object with metadata and an 'examples' list."
        )

    required = (
        "dataset_id",
        "version",
        "description",
        "labeling_guide_version",
        "examples",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise DatasetValidationError(
            "Dataset is missing required field(s): "
            f"{', '.join(missing)}."
        )
    for field in (
        "dataset_id",
        "version",
        "description",
        "labeling_guide_version",
    ):
        _nonempty_string(payload[field], field)

    raw_examples = payload["examples"]
    if not isinstance(raw_examples, list):
        raise DatasetValidationError("'examples' must be a list.")
    if not raw_examples:
        raise DatasetValidationError(
            "Dataset must contain at least one example."
        )

    seen_ids: set[str] = set()
    examples = tuple(
        _parse_example(raw, index, seen_ids)
        for index, raw in enumerate(raw_examples)
    )
    exclusions = _parse_exclusions(
        payload.get("exclusions"),
        example_ids=seen_ids,
    )
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in ("examples", "exclusions")
    }
    return examples, exclusions, metadata


def parse_dataset(
    payload: object,
    *,
    path: Path = Path("<memory>"),
    sha256: str = "",
) -> ClaimRankingDataset:
    """Parse and validate an in-memory dataset payload."""

    examples, exclusions, metadata = _parse_payload(payload)
    return ClaimRankingDataset(
        path=Path(path),
        sha256=sha256,
        metadata=MappingProxyType(metadata),
        examples=examples,
        exclusions=exclusions,
    )


def load_dataset(path: str | Path) -> ClaimRankingDataset:
    """Load and validate a UTF-8 JSON dataset from disk."""

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

    examples, exclusions, metadata = _parse_payload(payload)
    return ClaimRankingDataset(
        path=dataset_path,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        metadata=MappingProxyType(metadata),
        examples=examples,
        exclusions=exclusions,
    )


def _validate_objective(objective: object) -> str:
    value = getattr(objective, "value", objective)
    if not isinstance(value, str) or value not in OBJECTIVES:
        raise DatasetValidationError(
            f"objective must be one of {OBJECTIVES}; got {objective!r}."
        )
    return value


def baseline_ranking(
    dataset: ClaimRankingDataset,
    objective: str,
) -> BaselineEvaluationSet:
    """Build one objective's baseline ranking and evaluation labels.

    Claims with a null rank for the requested objective are excluded and listed.
    Ranks sort ascending. Duplicate ranks are allowed for evaluation fixtures and
    are resolved deterministically by dataset order, matching the production
    prioritizer's stable tie behavior.
    """

    selected = _validate_objective(objective)
    eligible: list[tuple[int, ClaimRankingExample]] = []
    exclusions: list[Exclusion] = []

    for index, example in enumerate(dataset.examples):
        rank = example.baseline_ranks[selected]
        if rank is None:
            exclusions.append(
                Exclusion(
                    claim_id=example.claim_id,
                    reason=f"No baseline rank for {selected}.",
                )
            )
        else:
            eligible.append((index, example))

    claim_ids = tuple(example.claim_id for _, example in eligible)
    ranked = sorted(
        eligible,
        key=lambda item: (
            item[1].baseline_ranks[selected],
            item[0],
        ),
    )
    ranked_claim_ids = tuple(example.claim_id for _, example in ranked)
    relevance = MappingProxyType(
        {
            example.claim_id: example.relevance_grade
            for _, example in eligible
        }
    )
    return BaselineEvaluationSet(
        objective=selected,
        claim_ids=claim_ids,
        ranked_claim_ids=ranked_claim_ids,
        relevance=relevance,
        exclusions=tuple(exclusions),
    )


def dataset_summary(dataset: ClaimRankingDataset) -> dict[str, Any]:
    """Return dataset readiness counts without computing any metrics."""

    grade_counts = {
        str(grade): 0
        for grade in range(MAX_RELEVANCE_GRADE + 1)
    }
    for example in dataset.examples:
        grade_counts[str(example.relevance_grade)] += 1

    baseline_rank_counts = {
        objective: sum(
            1
            for example in dataset.examples
            if example.baseline_ranks[objective] is not None
        )
        for objective in OBJECTIVES
    }
    return {
        "num_examples": len(dataset.examples),
        "relevance_grade_counts": grade_counts,
        "missing_ecs_count": sum(
            1
            for example in dataset.examples
            if example.ecs is None
        ),
        "baseline_rank_counts": baseline_rank_counts,
        "missing_baseline_rank_counts": {
            objective: len(dataset.examples) - count
            for objective, count in baseline_rank_counts.items()
        },
        "dataset_exclusions": [
            {
                "claim_id": exclusion.claim_id,
                "reason": exclusion.reason,
            }
            for exclusion in dataset.exclusions
        ],
    }


def evaluate_baseline(
    dataset: ClaimRankingDataset,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    *,
    relevant_min_grade: int = DEFAULT_RELEVANT_MIN_GRADE,
) -> dict[str, Any]:
    """Compute baseline metrics for all three objective-specific rankings.

    Each objective is evaluated on the claims that have a non-null baseline rank
    for that objective. Claims with a null rank are excluded and listed with a
    reason. The same human labels are used for every objective.
    """

    validated_k_values = validate_k_values(k_values)
    objectives: dict[str, Any] = {}

    for objective in OBJECTIVES:
        evaluation_set = baseline_ranking(dataset, objective)
        metrics = evaluate_ranking(
            evaluation_set.ranked_claim_ids,
            evaluation_set.relevance,
            validated_k_values,
            relevant_min_grade=relevant_min_grade,
        )
        objectives[objective] = {
            "num_claims": len(evaluation_set.claim_ids),
            "num_ranked": len(evaluation_set.ranked_claim_ids),
            "exclusions": [
                {
                    "claim_id": exclusion.claim_id,
                    "reason": exclusion.reason,
                }
                for exclusion in evaluation_set.exclusions
            ],
            "metrics": {
                str(k): {
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "ndcg": values["ndcg"],
                }
                for k, values in metrics.items()
            },
        }

    return {
        "objectives": objectives,
        "k_values": list(validated_k_values),
        "relevant_min_grade": relevant_min_grade,
        "dataset_exclusions": [
            {
                "claim_id": exclusion.claim_id,
                "reason": exclusion.reason,
            }
            for exclusion in dataset.exclusions
        ],
    }


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "DEFAULT_DATASET_ID",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_DATASET_VERSION",
    "DEFAULT_LABELING_GUIDE_VERSION",
    "OBJECTIVES",
    "BaselineEvaluationSet",
    "ClaimRankingDataset",
    "ClaimRankingExample",
    "DatasetValidationError",
    "EvidenceRecord",
    "Exclusion",
    "baseline_ranking",
    "dataset_summary",
    "evaluate_baseline",
    "load_dataset",
    "parse_dataset",
]
