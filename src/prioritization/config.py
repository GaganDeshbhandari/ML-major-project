"""Configuration for locally stored prioritization model artifacts.

The public prioritization API never accepts paths. Deployment configuration may
override the project-relative defaults through environment variables.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "models" / "prioritizer"

DEFAULT_O1_MODEL_FILENAME = "random_forest_model_o1.pkl"
DEFAULT_O2_MODEL_FILENAME = "random_forest_model.pkl"
DEFAULT_SCALER_FILENAME = "scaler.pkl"

ENV_ARTIFACT_DIR = "PRIORITIZATION_ARTIFACT_DIR"
ENV_O1_MODEL_PATH = "PRIORITIZATION_O1_MODEL_PATH"
ENV_O2_MODEL_PATH = "PRIORITIZATION_O2_MODEL_PATH"
ENV_SCALER_PATH = "PRIORITIZATION_SCALER_PATH"


def _env_value(env: Mapping[str, str], key: str) -> str | None:
    """Return a non-empty environment value, treating whitespace as unset."""

    value = env.get(key)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _resolve_local_path(value: str | os.PathLike[str], env_name: str) -> Path:
    """Resolve a trusted local path relative to the repository root.

    URL-like values are rejected so artifacts cannot be loaded from a remote
    location through environment configuration.
    """

    text = os.fspath(value)
    if "://" in text:
        raise ConfigurationError(
            f"{env_name} must be a local file-system path, not a URL: {text!r}."
        )

    path = Path(text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    """Local paths to the three trusted prioritization artifacts."""

    o1_model: Path
    o2_model: Path
    scaler: Path

    @classmethod
    def default(cls) -> ArtifactPaths:
        """Return the project-relative default artifact paths."""

        return cls(
            o1_model=(DEFAULT_ARTIFACT_DIR / DEFAULT_O1_MODEL_FILENAME).resolve(),
            o2_model=(DEFAULT_ARTIFACT_DIR / DEFAULT_O2_MODEL_FILENAME).resolve(),
            scaler=(DEFAULT_ARTIFACT_DIR / DEFAULT_SCALER_FILENAME).resolve(),
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ArtifactPaths:
        """Build artifact paths from optional environment overrides.

        Precedence is: file-specific override, artifact directory override,
        project-relative default. Relative overrides are resolved against the
        repository root.
        """

        values = os.environ if env is None else env

        raw_dir = _env_value(values, ENV_ARTIFACT_DIR)
        artifact_dir = (
            _resolve_local_path(raw_dir, ENV_ARTIFACT_DIR)
            if raw_dir is not None
            else DEFAULT_ARTIFACT_DIR.resolve()
        )

        def resolve_file(env_name: str, filename: str) -> Path:
            raw_path = _env_value(values, env_name)
            if raw_path is not None:
                return _resolve_local_path(raw_path, env_name)
            return (artifact_dir / filename).resolve()

        return cls(
            o1_model=resolve_file(ENV_O1_MODEL_PATH, DEFAULT_O1_MODEL_FILENAME),
            o2_model=resolve_file(ENV_O2_MODEL_PATH, DEFAULT_O2_MODEL_FILENAME),
            scaler=resolve_file(ENV_SCALER_PATH, DEFAULT_SCALER_FILENAME),
        )
