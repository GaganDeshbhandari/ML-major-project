"""Tests for immutable ECS configuration and environment parsing."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from src.evidence_conflict import ECSConfig
from src.evidence_conflict.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_MAX_EVIDENCE_CHARS,
    DEFAULT_MAX_LENGTH,
    DEFAULT_NLI_MODEL,
    ENV_BATCH_SIZE,
    ENV_CONFIDENCE_THRESHOLD,
    ENV_DEVICE,
    ENV_LOCAL_FILES_ONLY,
    ENV_MAX_EVIDENCE_CHARS,
    ENV_MAX_LENGTH,
    ENV_MODEL_REVISION,
    ENV_NLI_MODEL,
    MAX_SEQUENCE_LENGTH,
)
from src.evidence_conflict.errors import ECSConfigurationError

ALL_ENV_KEYS = (
    ENV_NLI_MODEL,
    ENV_DEVICE,
    ENV_MAX_LENGTH,
    ENV_BATCH_SIZE,
    ENV_CONFIDENCE_THRESHOLD,
    ENV_MAX_EVIDENCE_CHARS,
    ENV_LOCAL_FILES_ONLY,
    ENV_MODEL_REVISION,
)


def test_default_config() -> None:
    config = ECSConfig()

    assert config.model_name == DEFAULT_NLI_MODEL
    assert config.model_name == "cross-encoder/nli-deberta-v3-xsmall"
    assert config.device == DEFAULT_DEVICE == "auto"
    assert config.max_length == DEFAULT_MAX_LENGTH == 512
    assert config.batch_size == DEFAULT_BATCH_SIZE == 8
    assert config.confidence_threshold is None
    assert config.max_evidence_chars == DEFAULT_MAX_EVIDENCE_CHARS == 2000
    assert config.local_files_only is False
    assert config.model_revision is None


def test_config_is_frozen() -> None:
    config = ECSConfig()

    with pytest.raises(FrozenInstanceError):
        config.batch_size = 4  # type: ignore[misc]


def test_valid_custom_config_normalizes_strings() -> None:
    config = ECSConfig(
        model_name="  local/model  ",
        device=" CUDA:0 ",
        max_length=256,
        batch_size=4,
        confidence_threshold=0.6,
        max_evidence_chars=1000,
        local_files_only=True,
        model_revision="  revision-1  ",
    )

    assert config.model_name == "local/model"
    assert config.device == "cuda:0"
    assert config.max_length == 256
    assert config.batch_size == 4
    assert config.confidence_threshold == 0.6
    assert config.max_evidence_chars == 1000
    assert config.local_files_only is True
    assert config.model_revision == "revision-1"


@pytest.mark.parametrize(
    ("raw_device", "expected"),
    [
        ("auto", "auto"),
        ("CPU", "cpu"),
        ("cuda", "cuda"),
        ("CUDA:1", "cuda:1"),
    ],
)
def test_valid_device_values(raw_device: str, expected: str) -> None:
    assert ECSConfig(device=raw_device).device == expected


@pytest.mark.parametrize(
    "device",
    ["gpu", "cuda:", "cuda:-1", "cuda:abc", "", "   ", None, 0],
)
def test_invalid_device_values(device: object) -> None:
    with pytest.raises(ECSConfigurationError):
        ECSConfig(device=device)  # type: ignore[arg-type]


@pytest.mark.parametrize("model_name", ["", "   ", None, 123])
def test_invalid_model_name(model_name: object) -> None:
    with pytest.raises(ECSConfigurationError):
        ECSConfig(model_name=model_name)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "max_length",
    [0, -1, MAX_SEQUENCE_LENGTH + 1, True, 1.5, "512"],
)
def test_invalid_max_length(max_length: object) -> None:
    with pytest.raises(ECSConfigurationError):
        ECSConfig(max_length=max_length)  # type: ignore[arg-type]


def test_max_length_boundaries_are_valid() -> None:
    assert ECSConfig(max_length=1).max_length == 1
    assert (
        ECSConfig(max_length=MAX_SEQUENCE_LENGTH).max_length
        == MAX_SEQUENCE_LENGTH
    )


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, "8"])
def test_invalid_batch_size(batch_size: object) -> None:
    with pytest.raises(ECSConfigurationError):
        ECSConfig(batch_size=batch_size)  # type: ignore[arg-type]


def test_batch_size_one_is_valid() -> None:
    assert ECSConfig(batch_size=1).batch_size == 1


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1, math.nan, math.inf, -math.inf, True, "0.5"],
)
def test_invalid_confidence_threshold(threshold: object) -> None:
    with pytest.raises(ECSConfigurationError):
        ECSConfig(confidence_threshold=threshold)  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_confidence_threshold_boundaries_are_valid(
    threshold: float,
) -> None:
    assert ECSConfig(confidence_threshold=threshold).confidence_threshold == threshold


@pytest.mark.parametrize(
    "max_evidence_chars",
    [0, -1, True, 1.5, "2000"],
)
def test_invalid_max_evidence_chars(max_evidence_chars: object) -> None:
    with pytest.raises(ECSConfigurationError):
        ECSConfig(  # type: ignore[arg-type]
            max_evidence_chars=max_evidence_chars
        )


def test_max_evidence_chars_one_is_valid() -> None:
    assert ECSConfig(max_evidence_chars=1).max_evidence_chars == 1


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_local_files_only_requires_bool(value: object) -> None:
    with pytest.raises(ECSConfigurationError):
        ECSConfig(local_files_only=value)  # type: ignore[arg-type]


def test_model_revision_normalization() -> None:
    assert ECSConfig(model_revision=None).model_revision is None
    assert ECSConfig(model_revision="  main  ").model_revision == "main"
    assert ECSConfig(model_revision="   ").model_revision is None

    with pytest.raises(ECSConfigurationError):
        ECSConfig(model_revision=123)  # type: ignore[arg-type]


def test_from_env_defaults_to_constructor_defaults() -> None:
    assert ECSConfig.from_env({}) == ECSConfig()


def test_from_env_parses_all_values() -> None:
    config = ECSConfig.from_env(
        {
            ENV_NLI_MODEL: "custom/model",
            ENV_DEVICE: "CUDA:0",
            ENV_MAX_LENGTH: "256",
            ENV_BATCH_SIZE: "4",
            ENV_CONFIDENCE_THRESHOLD: "0.6",
            ENV_MAX_EVIDENCE_CHARS: "1000",
            ENV_LOCAL_FILES_ONLY: "yes",
            ENV_MODEL_REVISION: "  rev-1  ",
        }
    )

    assert config.model_name == "custom/model"
    assert config.device == "cuda:0"
    assert config.max_length == 256
    assert config.batch_size == 4
    assert config.confidence_threshold == 0.6
    assert config.max_evidence_chars == 1000
    assert config.local_files_only is True
    assert config.model_revision == "rev-1"


def test_from_env_blank_values_use_defaults() -> None:
    env = {key: "   " for key in ALL_ENV_KEYS}

    assert ECSConfig.from_env(env) == ECSConfig()


@pytest.mark.parametrize(
    "env",
    [
        {ENV_MAX_LENGTH: "abc"},
        {ENV_BATCH_SIZE: "0"},
        {ENV_MAX_EVIDENCE_CHARS: "-1"},
    ],
)
def test_from_env_invalid_integer_values(env: dict[str, str]) -> None:
    with pytest.raises(ECSConfigurationError):
        ECSConfig.from_env(env)


@pytest.mark.parametrize(
    "env",
    [
        {ENV_CONFIDENCE_THRESHOLD: "abc"},
        {ENV_CONFIDENCE_THRESHOLD: "nan"},
        {ENV_CONFIDENCE_THRESHOLD: "1.1"},
    ],
)
def test_from_env_invalid_float_values(env: dict[str, str]) -> None:
    with pytest.raises(ECSConfigurationError):
        ECSConfig.from_env(env)


@pytest.mark.parametrize(
    "env",
    [
        {ENV_LOCAL_FILES_ONLY: "maybe"},
        {ENV_DEVICE: "gpu"},
    ],
)
def test_from_env_invalid_boolean_or_device(
    env: dict[str, str],
) -> None:
    with pytest.raises(ECSConfigurationError):
        ECSConfig.from_env(env)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("NO", False),
        ("off", False),
    ],
)
def test_from_env_boolean_variants(raw: str, expected: bool) -> None:
    config = ECSConfig.from_env({ENV_LOCAL_FILES_ONLY: raw})

    assert config.local_files_only is expected


def test_from_env_uses_provided_mapping_only(monkeypatch) -> None:
    monkeypatch.setenv(ENV_BATCH_SIZE, "99")

    assert ECSConfig.from_env({}).batch_size == DEFAULT_BATCH_SIZE


def test_from_env_reads_os_environ_when_mapping_is_omitted(
    monkeypatch,
) -> None:
    for key in ALL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(ENV_BATCH_SIZE, "3")

    assert ECSConfig.from_env().batch_size == 3


def test_from_env_does_not_mutate_mapping() -> None:
    env = {ENV_BATCH_SIZE: "4", ENV_NLI_MODEL: "custom/model"}

    ECSConfig.from_env(env)

    assert env == {ENV_BATCH_SIZE: "4", ENV_NLI_MODEL: "custom/model"}


def test_from_env_rejects_non_mapping() -> None:
    with pytest.raises(ECSConfigurationError):
        ECSConfig.from_env([])  # type: ignore[arg-type]


def test_from_env_rejects_non_string_value() -> None:
    with pytest.raises(ECSConfigurationError):
        ECSConfig.from_env({ENV_BATCH_SIZE: 4})  # type: ignore[dict-item]
