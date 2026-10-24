"""Statistical feature extraction used by the O1 and O2 models."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import skew

from ..errors import InputValidationError
from ..preprocessing.ratings import map_rating_to_ptl

FEATURE_COLUMNS: tuple[str, ...] = ("mean", "median", "variance", "skewness")


@dataclass(frozen=True, slots=True)
class ClaimFeatures:
    """The four rating-derived features expected by the saved scaler."""

    mean: float
    median: float
    variance: float
    skewness: float

    def to_vector(self) -> np.ndarray:
        """Return features in the exact order used during training."""

        return np.asarray(
            [self.mean, self.median, self.variance, self.skewness],
            dtype=np.float64,
        )


def _as_ptl_values(ptl_values: Sequence[float] | np.ndarray) -> np.ndarray:
    try:
        values = np.asarray(list(ptl_values), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputValidationError("PTL values must be a sequence of numbers.") from exc

    if values.ndim != 1:
        raise InputValidationError("PTL values must be a one-dimensional sequence.")
    if values.size == 0:
        raise InputValidationError("Cannot compute features without PTL values.")
    if not np.all(np.isfinite(values)):
        raise InputValidationError("PTL values must all be finite numbers.")
    return values


def _sample_variance(values: np.ndarray) -> float:
    """Return pandas-compatible sample variance (ddof=1)."""

    if values.size < 2:
        return math.nan
    return float(np.var(values, ddof=1))


def _zero_if_nan(value: float) -> float:
    return 0.0 if math.isnan(value) else value


def compute_claim_features(ptl_values: Sequence[float] | np.ndarray) -> ClaimFeatures:
    """Compute mean, median, variance, and skewness for one claim.

    Undefined feature values are replaced with ``0`` as in the notebook's
    ``fillna(0)`` step. This function never scales or fits anything.
    """

    values = _as_ptl_values(ptl_values)

    mean = float(np.mean(values))
    median = float(np.median(values))
    variance = _sample_variance(values)
    skewness = float(skew(values))

    return ClaimFeatures(
        mean=_zero_if_nan(mean),
        median=_zero_if_nan(median),
        variance=_zero_if_nan(variance),
        skewness=_zero_if_nan(skewness),
    )


def extract_claim_features(ratings: Sequence[str]) -> ClaimFeatures:
    """Map rating labels to PTL and compute one claim's features."""

    return compute_claim_features([map_rating_to_ptl(rating) for rating in ratings])


def compute_disputability(ptl_values: Sequence[float] | np.ndarray) -> float:
    """Return sample variance (``ddof=1``) for O3.

    Unlike O1/O2 feature extraction, undefined variance is not replaced with
    zero. A claim with fewer than two ratings returns ``NaN``.
    """

    values = _as_ptl_values(ptl_values)
    return _sample_variance(values)
