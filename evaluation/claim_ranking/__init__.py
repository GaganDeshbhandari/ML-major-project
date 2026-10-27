"""Offline evaluation tooling for claim-ranking methods.

This package validates manually labeled claim-ranking datasets, computes
Precision@K, Recall@K, and NDCG@K, and builds reproducible per-objective
baseline and RRF fusion comparison reports. The fusion CLI is available as
``evaluation.claim_ranking.fusion``. This package does not change production
prioritization, retrieval, ECS, or FactLLaMA behavior.
"""

from .dataset import (
    DATASET_SCHEMA_VERSION,
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_PATH,
    DEFAULT_DATASET_VERSION,
    DEFAULT_LABELING_GUIDE_VERSION,
    OBJECTIVES,
    BaselineEvaluationSet,
    ClaimRankingDataset,
    ClaimRankingExample,
    DatasetValidationError,
    EvidenceRecord,
    Exclusion,
    baseline_ranking,
    dataset_summary,
    evaluate_baseline,
    load_dataset,
    parse_dataset,
)
from .metrics import (
    DEFAULT_K_VALUES,
    DEFAULT_RELEVANT_MIN_GRADE,
    MAX_RELEVANCE_GRADE,
    EvaluationInputError,
    evaluate_ranking,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    stable_rank_by_score,
    validate_k,
    validate_k_values,
    validate_relevant_min_grade,
)

__all__ = [
    "DATASET_SCHEMA_VERSION",
    "DEFAULT_DATASET_ID",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_DATASET_VERSION",
    "DEFAULT_K_VALUES",
    "DEFAULT_LABELING_GUIDE_VERSION",
    "DEFAULT_RELEVANT_MIN_GRADE",
    "MAX_RELEVANCE_GRADE",
    "OBJECTIVES",
    "BaselineEvaluationSet",
    "ClaimRankingDataset",
    "ClaimRankingExample",
    "DatasetValidationError",
    "EvaluationInputError",
    "EvidenceRecord",
    "Exclusion",
    "baseline_ranking",
    "dataset_summary",
    "evaluate_baseline",
    "evaluate_ranking",
    "load_dataset",
    "ndcg_at_k",
    "parse_dataset",
    "precision_at_k",
    "recall_at_k",
    "stable_rank_by_score",
    "validate_k",
    "validate_k_values",
    "validate_relevant_min_grade",
]
