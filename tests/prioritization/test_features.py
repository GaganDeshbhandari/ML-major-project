"""Tests for statistical feature extraction and disputability."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.prioritization.errors import InputValidationError
from src.prioritization.features.extraction import (
    FEATURE_COLUMNS,
    compute_claim_features,
    compute_disputability,
    extract_claim_features,
)


def test_feature_order_is_exact() -> None:
    assert FEATURE_COLUMNS == ("mean", "median", "variance", "skewness")
    vector = compute_claim_features([1.0, 0.0, -1.0]).to_vector()
    assert vector.shape == (4,)
    assert vector.tolist() == pytest.approx([0.0, 0.0, 1.0, 0.0])


def test_feature_values_match_notebook_math() -> None:
    features = extract_claim_features(
        ["True", "True", "Mostly True", "Mixture", "False"]
    )

    assert features.mean == pytest.approx(0.3)
    assert features.median == pytest.approx(0.5)
    assert features.variance == pytest.approx(0.7)
    assert features.skewness == pytest.approx(-0.7301958930821597)


def test_feature_vector_preserves_column_order() -> None:
    features = compute_claim_features([1.0, 1.0, 0.5, 0.0, -1.0])
    np.testing.assert_allclose(
        features.to_vector(),
        np.asarray(
            [features.mean, features.median, features.variance, features.skewness],
            dtype=np.float64,
        ),
    )


def test_constant_ratings_replace_missing_skewness_with_zero() -> None:
    features = compute_claim_features([0.5, 0.5, 0.5])

    assert features.mean == pytest.approx(0.5)
    assert features.median == pytest.approx(0.5)
    assert features.variance == pytest.approx(0.0)
    assert features.skewness == 0.0


def test_single_rating_has_zero_filled_features_except_location() -> None:
    features = compute_claim_features([0.5])

    assert features.mean == pytest.approx(0.5)
    assert features.median == pytest.approx(0.5)
    assert features.variance == 0.0
    assert features.skewness == 0.0


def test_disputability_uses_sample_variance() -> None:
    # Population variance would be 8/9; pandas GroupBy.var() uses ddof=1.
    assert compute_disputability([1.0, -1.0, 1.0]) == pytest.approx(4.0 / 3.0)
    assert compute_disputability([1.0, -1.0]) == pytest.approx(2.0)
    assert compute_disputability([1.0, 1.0, 1.0]) == pytest.approx(0.0)


def test_single_rating_disputability_is_undefined() -> None:
    assert math.isnan(compute_disputability([1.0]))


def test_empty_features_are_rejected() -> None:
    with pytest.raises(InputValidationError):
        compute_claim_features([])
    with pytest.raises(InputValidationError):
        compute_disputability([])


def test_nonfinite_ptl_values_are_rejected() -> None:
    with pytest.raises(InputValidationError):
        compute_claim_features([1.0, math.nan])
    with pytest.raises(InputValidationError):
        compute_disputability([1.0, math.inf])
