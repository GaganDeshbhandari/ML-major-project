"""Tests for the evidence conflict exception hierarchy."""

from __future__ import annotations

import pytest

from src.evidence_conflict.errors import (
    ECSConfigurationError,
    ECSError,
    EvidenceConflictError,
    StanceInferenceError,
    StanceInputError,
    StanceModelError,
    UnsupportedLabelMappingError,
)


def test_all_errors_inherit_from_the_feature_base() -> None:
    error_types = [
        ECSConfigurationError,
        ECSError,
        StanceInferenceError,
        StanceInputError,
        StanceModelError,
        UnsupportedLabelMappingError,
    ]

    for error_type in error_types:
        assert issubclass(error_type, EvidenceConflictError)


def test_input_and_configuration_errors_are_value_errors() -> None:
    assert issubclass(StanceInputError, ValueError)
    assert issubclass(ECSConfigurationError, ValueError)


def test_runtime_errors_are_runtime_errors() -> None:
    assert issubclass(StanceModelError, RuntimeError)
    assert issubclass(StanceInferenceError, RuntimeError)
    assert issubclass(ECSError, RuntimeError)


def test_label_mapping_error_is_a_model_error() -> None:
    assert issubclass(UnsupportedLabelMappingError, StanceModelError)


def test_feature_error_can_be_caught_as_the_base_type() -> None:
    with pytest.raises(EvidenceConflictError):
        raise StanceInputError("invalid stance input")
