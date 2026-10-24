"""Bias-aware news prioritization.

Phase 1 implements the three notebook objectives:

* O1: predict and return claims likely to be ``False``.
* O2: predict and return claims with ``High`` total perception bias.
* O3: rank claims by descending PTL sample variance (disputability).

Predictions are model outputs, not verified facts.
"""

from .config import DEFAULT_ARTIFACT_DIR, ArtifactPaths
from .errors import (
    ArtifactError,
    ArtifactLoadError,
    ArtifactNotFoundError,
    ArtifactValidationError,
    ConfigurationError,
    DuplicateClaimError,
    InferenceError,
    InputValidationError,
    InvalidClaimError,
    InvalidObjectiveError,
    InvalidRatingError,
    MissingPerceptionError,
    PrioritizationError,
    UnknownClaimError,
)
from .features.extraction import (
    FEATURE_COLUMNS,
    ClaimFeatures,
    compute_claim_features,
    compute_disputability,
    extract_claim_features,
)
from .inference.artifacts import (
    ArtifactStore,
    clear_default_artifact_store_cache,
    get_default_artifact_store,
)
from .inference.prioritizer import prioritize_claims
from .preprocessing.ratings import (
    PTL_MAPPING,
    PreparedRatings,
    map_rating_to_ptl,
    prepare_ratings,
)
from .schemas import (
    Claim,
    ClaimId,
    Objective,
    PrioritizedClaim,
    TPBLabel,
    TruthLabel,
    UserRating,
)

__all__ = [
    "DEFAULT_ARTIFACT_DIR",
    "FEATURE_COLUMNS",
    "PTL_MAPPING",
    "ArtifactError",
    "ArtifactLoadError",
    "ArtifactNotFoundError",
    "ArtifactPaths",
    "ArtifactStore",
    "ArtifactValidationError",
    "Claim",
    "ClaimFeatures",
    "ClaimId",
    "ConfigurationError",
    "DuplicateClaimError",
    "InferenceError",
    "InputValidationError",
    "InvalidClaimError",
    "InvalidObjectiveError",
    "InvalidRatingError",
    "MissingPerceptionError",
    "Objective",
    "PreparedRatings",
    "PrioritizationError",
    "PrioritizedClaim",
    "TPBLabel",
    "TruthLabel",
    "UnknownClaimError",
    "UserRating",
    "clear_default_artifact_store_cache",
    "compute_claim_features",
    "compute_disputability",
    "extract_claim_features",
    "get_default_artifact_store",
    "map_rating_to_ptl",
    "prepare_ratings",
    "prioritize_claims",
]
