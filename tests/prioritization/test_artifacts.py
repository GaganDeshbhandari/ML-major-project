"""Tests for artifact configuration, loading, caching, and validation."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from src.prioritization.config import (
    DEFAULT_ARTIFACT_DIR,
    PROJECT_ROOT,
    ArtifactPaths,
)
from src.prioritization.errors import (
    ArtifactLoadError,
    ArtifactNotFoundError,
    ArtifactValidationError,
    ConfigurationError,
)
from src.prioritization.features.extraction import FEATURE_COLUMNS
from src.prioritization.inference.artifacts import (
    ArtifactStore,
    clear_default_artifact_store_cache,
    get_default_artifact_store,
)


class DummyModel:
    def __init__(self, classes: tuple[str, ...] = ("False", "True")) -> None:
        self.classes_ = np.asarray(classes, dtype=object)
        self.n_features_in_ = len(FEATURE_COLUMNS)

    def predict(self, X: object) -> np.ndarray:
        return np.asarray([str(self.classes_[0])] * len(X), dtype=object)  # type: ignore[arg-type]


class WrongCountModel(DummyModel):
    def __init__(self) -> None:
        super().__init__()
        self.n_features_in_ = 3


class DummyScaler:
    n_features_in_ = len(FEATURE_COLUMNS)
    feature_names_in_ = np.asarray(FEATURE_COLUMNS, dtype=object)

    def transform(self, X: object) -> np.ndarray:
        return np.asarray(X, dtype=np.float64)


class WrongOrderScaler(DummyScaler):
    feature_names_in_ = np.asarray(
        ("median", "mean", "variance", "skewness"), dtype=object
    )


class NoFeatureNamesScaler:
    n_features_in_ = len(FEATURE_COLUMNS)

    def transform(self, X: object) -> np.ndarray:
        return np.asarray(X, dtype=np.float64)


def _paths(tmp_path: Path) -> ArtifactPaths:
    return ArtifactPaths(
        o1_model=tmp_path / "random_forest_model_o1.pkl",
        o2_model=tmp_path / "random_forest_model.pkl",
        scaler=tmp_path / "scaler.pkl",
    )


def _write_bundle(
    tmp_path: Path,
    *,
    o1: object | None = None,
    o2: object | None = None,
    scaler: object | None = None,
) -> ArtifactPaths:
    paths = _paths(tmp_path)
    joblib.dump(o1 if o1 is not None else DummyModel(), paths.o1_model)
    joblib.dump(
        o2 if o2 is not None else DummyModel(classes=("High", "Low")),
        paths.o2_model,
    )
    joblib.dump(scaler if scaler is not None else DummyScaler(), paths.scaler)
    return paths


def test_loads_each_artifact_once(tmp_path: Path) -> None:
    paths = _write_bundle(tmp_path)
    calls: list[Path] = []

    def counting_loader(path: Path) -> object:
        calls.append(path)
        return joblib.load(path)

    store = ArtifactStore(paths, loader=counting_loader)

    assert store.o1_model is store.o1_model
    assert len(calls) == 1
    assert store.o2_model is store.o2_model
    assert len(calls) == 2
    assert store.scaler is store.scaler
    assert len(calls) == 3


def test_missing_o1_artifact_has_actionable_error(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    store = ArtifactStore(paths)

    with pytest.raises(ArtifactNotFoundError, match="random_forest_model_o1.pkl"):
        _ = store.o1_model


def test_missing_scaler_artifact_has_actionable_error(tmp_path: Path) -> None:
    paths = _write_bundle(tmp_path)
    paths.scaler.unlink()
    store = ArtifactStore(paths)

    with pytest.raises(ArtifactNotFoundError, match="scaler.pkl"):
        _ = store.scaler


def test_invalid_artifact_object_is_rejected(tmp_path: Path) -> None:
    paths = _write_bundle(tmp_path, o1={"not": "a model"})
    store = ArtifactStore(paths)

    with pytest.raises(ArtifactValidationError, match="predict"):
        _ = store.o1_model


def test_wrong_feature_count_is_rejected(tmp_path: Path) -> None:
    paths = _write_bundle(tmp_path, o1=WrongCountModel())
    store = ArtifactStore(paths)

    with pytest.raises(ArtifactValidationError, match="n_features_in_"):
        _ = store.o1_model


def test_swapped_model_classes_are_rejected(tmp_path: Path) -> None:
    paths = _write_bundle(tmp_path, o1=DummyModel(classes=("High", "Low")))
    store = ArtifactStore(paths)

    with pytest.raises(ArtifactValidationError, match="expected classes"):
        _ = store.o1_model


def test_wrong_scaler_feature_order_is_rejected(tmp_path: Path) -> None:
    paths = _write_bundle(tmp_path, scaler=WrongOrderScaler())
    store = ArtifactStore(paths)

    with pytest.raises(ArtifactValidationError, match="feature order"):
        _ = store.scaler


def test_scaler_without_feature_names_is_rejected(tmp_path: Path) -> None:
    paths = _write_bundle(tmp_path, scaler=NoFeatureNamesScaler())
    store = ArtifactStore(paths)

    with pytest.raises(ArtifactValidationError, match="feature_names_in_"):
        _ = store.scaler


def test_loader_failure_is_wrapped(tmp_path: Path) -> None:
    paths = _write_bundle(tmp_path)

    def failing_loader(path: Path) -> object:
        raise RuntimeError("unexpected deserialization failure")

    store = ArtifactStore(paths, loader=failing_loader)
    with pytest.raises(ArtifactLoadError, match="Could not load O1 model"):
        _ = store.o1_model


def test_default_store_is_cached(tmp_path: Path) -> None:
    clear_default_artifact_store_cache()
    paths = _write_bundle(tmp_path)

    first = get_default_artifact_store(paths)
    second = get_default_artifact_store(paths)
    assert first is second

    clear_default_artifact_store_cache()
    third = get_default_artifact_store(paths)
    assert third is not first


def test_default_paths_are_project_relative() -> None:
    paths = ArtifactPaths.default()

    assert paths.o1_model.name == "random_forest_model_o1.pkl"
    assert paths.o2_model.name == "random_forest_model.pkl"
    assert paths.scaler.name == "scaler.pkl"
    assert paths.o1_model.parent == DEFAULT_ARTIFACT_DIR


def test_environment_directory_override(tmp_path: Path) -> None:
    paths = ArtifactPaths.from_env({"PRIORITIZATION_ARTIFACT_DIR": str(tmp_path)})

    assert paths.o1_model == (tmp_path / "random_forest_model_o1.pkl").resolve()
    assert paths.o2_model == (tmp_path / "random_forest_model.pkl").resolve()
    assert paths.scaler == (tmp_path / "scaler.pkl").resolve()


def test_file_specific_environment_override_wins(tmp_path: Path) -> None:
    custom_o1 = tmp_path / "custom_o1.pkl"
    custom_o2 = tmp_path / "custom_o2.pkl"
    custom_scaler = tmp_path / "custom_scaler.pkl"
    paths = ArtifactPaths.from_env(
        {
            "PRIORITIZATION_ARTIFACT_DIR": str(tmp_path),
            "PRIORITIZATION_O1_MODEL_PATH": str(custom_o1),
            "PRIORITIZATION_O2_MODEL_PATH": str(custom_o2),
            "PRIORITIZATION_SCALER_PATH": str(custom_scaler),
        }
    )

    assert paths.o1_model == custom_o1.resolve()
    assert paths.o2_model == custom_o2.resolve()
    assert paths.scaler == custom_scaler.resolve()


def test_relative_environment_paths_resolve_against_project_root() -> None:
    paths = ArtifactPaths.from_env(
        {"PRIORITIZATION_ARTIFACT_DIR": "models/prioritizer"}
    )

    assert (
        paths.o1_model
        == (
            PROJECT_ROOT / "models" / "prioritizer" / "random_forest_model_o1.pkl"
        ).resolve()
    )


def test_url_artifact_configuration_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        ArtifactPaths.from_env(
            {"PRIORITIZATION_ARTIFACT_DIR": "https://example.com/models"}
        )
