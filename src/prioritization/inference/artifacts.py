"""Lazy loading, validation, and caching of prioritization artifacts."""

from __future__ import annotations

from collections.abc import Callable
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any

import joblib

from ..config import (
    ENV_O1_MODEL_PATH,
    ENV_O2_MODEL_PATH,
    ENV_SCALER_PATH,
    ArtifactPaths,
)
from ..errors import (
    ArtifactLoadError,
    ArtifactNotFoundError,
    ArtifactValidationError,
)
from ..features.extraction import FEATURE_COLUMNS

ArtifactLoader = Callable[[Path], Any]

O1_CLASSES = frozenset({"True", "False"})
O2_CLASSES = frozenset({"High", "Low"})


def _validation_error(role: str, path: Path, detail: str) -> ArtifactValidationError:
    return ArtifactValidationError(
        f"Invalid {role} artifact at '{path}': {detail} "
        "The artifact must match the feature contract used during training."
    )


def _validate_model(
    artifact: Any,
    *,
    role: str,
    path: Path,
    expected_classes: frozenset[str],
) -> None:
    if not callable(getattr(artifact, "predict", None)):
        raise _validation_error(role, path, "object has no callable predict().")

    feature_count = getattr(artifact, "n_features_in_", None)
    if feature_count != len(FEATURE_COLUMNS):
        raise _validation_error(
            role,
            path,
            f"expected n_features_in_={len(FEATURE_COLUMNS)}, found {feature_count!r}.",
        )

    classes = getattr(artifact, "classes_", None)
    if classes is None:
        raise _validation_error(role, path, "object has no classes_ attribute.")
    actual_classes = {str(label) for label in classes}
    if actual_classes != expected_classes:
        raise _validation_error(
            role,
            path,
            f"expected classes {sorted(expected_classes)}, "
            f"found {sorted(actual_classes)}.",
        )


def _validate_scaler(artifact: Any, *, path: Path) -> None:
    if not callable(getattr(artifact, "transform", None)):
        raise _validation_error("scaler", path, "object has no callable transform().")

    feature_count = getattr(artifact, "n_features_in_", None)
    if feature_count != len(FEATURE_COLUMNS):
        raise _validation_error(
            "scaler",
            path,
            f"expected n_features_in_={len(FEATURE_COLUMNS)}, found {feature_count!r}.",
        )

    feature_names = getattr(artifact, "feature_names_in_", None)
    if feature_names is None:
        raise _validation_error(
            "scaler",
            path,
            "object does not record feature_names_in_, so feature order "
            "cannot be verified.",
        )
    actual_names = tuple(str(name) for name in feature_names)
    if actual_names != FEATURE_COLUMNS:
        raise _validation_error(
            "scaler",
            path,
            f"expected feature order {FEATURE_COLUMNS}, found {actual_names}.",
        )


class ArtifactStore:
    """Lazy cache for trusted local prioritization artifacts.

    Each artifact is loaded at most once per store instance. O1 only touches
    the O1 model and scaler, O2 only touches the O2 model and scaler, and O3
    does not use a store at all.
    """

    def __init__(
        self,
        paths: ArtifactPaths | None = None,
        *,
        loader: ArtifactLoader = joblib.load,
    ) -> None:
        self._paths = paths if paths is not None else ArtifactPaths.from_env()
        self._loader = loader

    @property
    def paths(self) -> ArtifactPaths:
        """Return the configured local artifact paths."""

        return self._paths

    def _load(self, *, role: str, path: Path, env_name: str) -> Any:
        if not path.is_file():
            raise ArtifactNotFoundError(
                f"{role} artifact not found at '{path}'. Place '{path.name}' "
                f"in '{path.parent}' or set {env_name}."
            )

        try:
            artifact = self._loader(path)
        except Exception as exc:
            raise ArtifactLoadError(
                f"Could not load {role} artifact from '{path}'. Ensure the file "
                "is a trusted local artifact created with joblib.dump() and that "
                "the installed scikit-learn version is compatible with the "
                "version used to train it."
            ) from exc

        return artifact

    @cached_property
    def o1_model(self) -> Any:
        """Load and validate the O1 True/False classifier once."""

        path = self._paths.o1_model
        artifact = self._load(
            role="O1 model",
            path=path,
            env_name=ENV_O1_MODEL_PATH,
        )
        _validate_model(
            artifact,
            role="O1 model",
            path=path,
            expected_classes=O1_CLASSES,
        )
        return artifact

    @cached_property
    def o2_model(self) -> Any:
        """Load and validate the O2 High/Low classifier once."""

        path = self._paths.o2_model
        artifact = self._load(
            role="O2 model",
            path=path,
            env_name=ENV_O2_MODEL_PATH,
        )
        _validate_model(
            artifact,
            role="O2 model",
            path=path,
            expected_classes=O2_CLASSES,
        )
        return artifact

    @cached_property
    def scaler(self) -> Any:
        """Load and validate the shared StandardScaler once."""

        path = self._paths.scaler
        artifact = self._load(
            role="scaler",
            path=path,
            env_name=ENV_SCALER_PATH,
        )
        _validate_scaler(artifact, path=path)
        return artifact


@lru_cache(maxsize=8)
def _cached_default_store(paths: ArtifactPaths) -> ArtifactStore:
    return ArtifactStore(paths)


def get_default_artifact_store(paths: ArtifactPaths | None = None) -> ArtifactStore:
    """Return a process-wide cached artifact store for the configured paths."""

    return _cached_default_store(
        paths if paths is not None else ArtifactPaths.from_env()
    )


def clear_default_artifact_store_cache() -> None:
    """Clear the process-wide artifact store cache.

    This is intended for tests and controlled configuration reloads, not for
    normal inference paths.
    """

    _cached_default_store.cache_clear()
