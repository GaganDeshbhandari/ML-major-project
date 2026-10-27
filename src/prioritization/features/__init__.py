"""Feature extraction for bias-aware prioritization."""

from .extraction import (
    FEATURE_COLUMNS,
    ClaimFeatures,
    compute_claim_features,
    compute_disputability,
    extract_claim_features,
)

__all__ = [
    "FEATURE_COLUMNS",
    "ClaimFeatures",
    "compute_claim_features",
    "compute_disputability",
    "extract_claim_features",
]
