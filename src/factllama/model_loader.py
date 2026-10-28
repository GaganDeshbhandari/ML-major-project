"""Configuration and lazy loading for FactLLaMA inference.

Base model: ``meta-llama/Llama-3.2-3B-Instruct`` with a QLoRA adapter
trained in 4-bit. Both the base model and the adapter location are
configurable through ``configs/factllama.json`` and ``FACTLLAMA_*``
environment variables; no machine-specific path is hardcoded.

Design notes:
    * Importing this module never imports ``torch``, ``transformers``,
      ``peft`` or ``bitsandbytes`` and never touches model weights.
    * Dependencies are imported on the first :meth:`FactLLaMAModelLoader.load`
      call and the result is cached on the loader instance.
    * Relative adapter paths are resolved against the repository root, so
      the default ``models/factllama/...`` works regardless of the current
      working directory.
    * Failures are reported with specific exceptions rather than generic
      import or hub errors.

Real loading with the trained adapter was verified against the cached
Llama-3.2-3B-Instruct snapshot; see ``docs/factllama.md`` for the validation
details.
"""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "ADAPTER_CONFIG_FILENAME",
    "ADAPTER_WEIGHT_FILENAMES",
    "CONFIG_PATH_ENV",
    "DEFAULT_ADAPTER_PATH",
    "DEFAULT_BASE_MODEL",
    "DEFAULT_CONFIG_PATH",
    "AdapterNotFoundError",
    "ConfigurationError",
    "FactLLaMAConfig",
    "FactLLaMAError",
    "FactLLaMAModelLoader",
    "LoadedFactLLaMA",
    "MissingDependencyError",
    "ModelAccessError",
    "ModelLoadError",
    "QuantizationUnsupportedError",
    "load_config",
    "project_root",
    "resolve_adapter_path",
    "validate_adapter_path",
]

DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_ADAPTER_PATH = "models/factllama/factllama_final_adapter_v2"
DEFAULT_CONFIG_PATH = "configs/factllama.json"

#: Environment variable pointing at an alternative JSON config file.
CONFIG_PATH_ENV = "FACTLLAMA_CONFIG"

ADAPTER_CONFIG_FILENAME = "adapter_config.json"
ADAPTER_WEIGHT_FILENAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
)
TOKENIZER_CONFIG_FILENAME = "tokenizer_config.json"

_CONFIG_ENV_OVERRIDES = {
    "base_model": "FACTLLAMA_BASE_MODEL",
    "adapter_path": "FACTLLAMA_ADAPTER_PATH",
    "load_in_4bit": "FACTLLAMA_LOAD_IN_4BIT",
    "bnb_4bit_quant_type": "FACTLLAMA_BNB_4BIT_QUANT_TYPE",
    "bnb_4bit_compute_dtype": "FACTLLAMA_BNB_4BIT_COMPUTE_DTYPE",
    "bnb_4bit_use_double_quant": "FACTLLAMA_BNB_4BIT_USE_DOUBLE_QUANT",
    "device_map": "FACTLLAMA_DEVICE_MAP",
    "trust_remote_code": "FACTLLAMA_TRUST_REMOTE_CODE",
    "max_new_tokens": "FACTLLAMA_MAX_NEW_TOKENS",
    "do_sample": "FACTLLAMA_DO_SAMPLE",
    "temperature": "FACTLLAMA_TEMPERATURE",
    "max_evidence_tokens": "FACTLLAMA_MAX_EVIDENCE_TOKENS",
    "evidence_chars_per_token": "FACTLLAMA_EVIDENCE_CHARS_PER_TOKEN",
}

_ALLOWED_CONFIG_KEYS = frozenset(_CONFIG_ENV_OVERRIDES)
_ALLOWED_QUANT_TYPES = ("nf4", "fp4")

_DEPENDENCY_HINTS = {
    "torch": "pip install 'torch>=2.2'",
    "transformers": "pip install 'transformers>=4.43'",
    "peft": "pip install 'peft>=0.11'",
    "accelerate": "pip install 'accelerate>=0.30'",
    "bitsandbytes": "pip install 'bitsandbytes>=0.43'",
}

_ACCESS_ERROR_MARKERS = (
    "gated",
    "authenticat",
    "authoriz",
    "permission",
    "401",
    "403",
    "token",
    "not a valid model identifier",
    "repository not found",
    "couldn't connect",
    "connection",
    "offline",
    "huggingface.co",
)


class FactLLaMAError(Exception):
    """Base class for FactLLaMA inference errors."""


class ConfigurationError(FactLLaMAError):
    """Raised when the FactLLaMA configuration is missing or invalid."""


class AdapterNotFoundError(FactLLaMAError):
    """Raised when the LoRA adapter folder or its essential files are absent."""


class MissingDependencyError(FactLLaMAError):
    """Raised when an optional inference dependency is not installed."""


class QuantizationUnsupportedError(FactLLaMAError):
    """Raised when 4-bit loading is requested without usable GPU support."""


class ModelAccessError(FactLLaMAError):
    """Raised when the base model or adapter cannot be accessed or downloaded."""


class ModelLoadError(FactLLaMAError):
    """Raised when model/tokenizer construction fails for other reasons."""


@dataclass(frozen=True)
class FactLLaMAConfig:
    """Runtime configuration for FactLLaMA inference.

    ``adapter_path`` may be absolute, project-relative or contain ``~``.
    Relative paths are resolved against :func:`project_root` at load time.
    ``max_evidence_tokens`` and ``evidence_chars_per_token`` size the
    approximate evidence budget used by the predictor.
    """

    base_model: str = DEFAULT_BASE_MODEL
    adapter_path: Path = field(default_factory=lambda: Path(DEFAULT_ADAPTER_PATH))
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    device_map: str = "auto"
    trust_remote_code: bool = False
    max_new_tokens: int = 16
    do_sample: bool = False
    temperature: float = 0.0
    max_evidence_tokens: int = 1024
    evidence_chars_per_token: float = 4.0


@dataclass(frozen=True)
class LoadedFactLLaMA:
    """A loaded base model + adapter + tokenizer bundle."""

    model: Any
    tokenizer: Any
    config: FactLLaMAConfig
    adapter_path: Path


def project_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def resolve_adapter_path(
    path: str | Path, root: str | Path | None = None
) -> Path:
    """Resolve an adapter path against the repository root.

    Absolute paths and ``~`` are honoured; everything else is treated as
    project-relative.
    """
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded
    base = Path(root) if root is not None else project_root()
    return (base / expanded).resolve()


def validate_adapter_path(path: str | Path) -> Path:
    """Validate an adapter folder and return its resolved path.

    An adapter is considered complete when ``adapter_config.json`` and one
    of ``adapter_model.safetensors`` / ``adapter_model.bin`` are present.
    Tokenizer files are optional: when absent the base model tokenizer is
    used.
    """
    resolved = resolve_adapter_path(path)

    if not resolved.exists():
        raise AdapterNotFoundError(
            f"Adapter directory not found: {resolved}. Copy the "
            "'factllama_final_adapter_v2' folder from Google Drive into "
            "models/factllama/ or point FACTLLAMA_ADAPTER_PATH at it."
        )
    if not resolved.is_dir():
        raise AdapterNotFoundError(
            f"Adapter path is not a directory: {resolved}"
        )

    missing: list[str] = []
    if not (resolved / ADAPTER_CONFIG_FILENAME).is_file():
        missing.append(ADAPTER_CONFIG_FILENAME)
    if not any(
        (resolved / name).is_file() for name in ADAPTER_WEIGHT_FILENAMES
    ):
        missing.append(" or ".join(ADAPTER_WEIGHT_FILENAMES))

    if missing:
        raise AdapterNotFoundError(
            f"Adapter folder {resolved} is incomplete; missing: "
            f"{', '.join(missing)}. Expected {ADAPTER_CONFIG_FILENAME}, "
            "adapter_model.safetensors and tokenizer files."
        )

    return resolved


def _as_str(value: Any, key: str) -> str:
    if isinstance(value, Path):
        value = str(value)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{key} must be a non-empty string")
    return value.strip()


def _as_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ConfigurationError(
        f"{key} must be a boolean (true/false), got {value!r}"
    )


def _as_int(value: Any, key: str) -> int:
    if isinstance(value, bool):
        raise ConfigurationError(f"{key} must be an integer, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{key} must be an integer, got {value!r}"
        ) from exc
    return parsed


def _as_float(value: Any, key: str) -> float:
    if isinstance(value, bool):
        raise ConfigurationError(f"{key} must be a number, got {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{key} must be a number, got {value!r}"
        ) from exc
    return parsed


def _build_config(
    data: Mapping[str, Any],
    config_path: Path,
    env: Mapping[str, str],
) -> FactLLaMAConfig:
    merged: dict[str, Any] = dict(data)

    for key, env_name in _CONFIG_ENV_OVERRIDES.items():
        value = env.get(env_name)
        if value is not None and str(value).strip() != "":
            merged[key] = value

    unknown = sorted(set(merged) - _ALLOWED_CONFIG_KEYS)
    if unknown:
        raise ConfigurationError(
            f"Unknown FactLLaMA config key(s) in {config_path}: "
            f"{', '.join(unknown)}"
        )

    base_model = _as_str(merged.get("base_model", DEFAULT_BASE_MODEL), "base_model")
    adapter_path = _as_str(
        merged.get("adapter_path", DEFAULT_ADAPTER_PATH), "adapter_path"
    )
    quant_type = _as_str(
        merged.get("bnb_4bit_quant_type", "nf4"), "bnb_4bit_quant_type"
    ).lower()
    if quant_type not in _ALLOWED_QUANT_TYPES:
        raise ConfigurationError(
            "bnb_4bit_quant_type must be one of "
            f"{', '.join(_ALLOWED_QUANT_TYPES)}, got {quant_type!r}"
        )

    max_new_tokens = _as_int(
        merged.get("max_new_tokens", 16), "max_new_tokens"
    )
    if max_new_tokens < 1:
        raise ConfigurationError("max_new_tokens must be >= 1")

    temperature = _as_float(merged.get("temperature", 0.0), "temperature")
    if temperature < 0:
        raise ConfigurationError("temperature must be >= 0")

    max_evidence_tokens = _as_int(
        merged.get("max_evidence_tokens", 1024), "max_evidence_tokens"
    )
    if max_evidence_tokens < 1:
        raise ConfigurationError("max_evidence_tokens must be >= 1")

    evidence_chars_per_token = _as_float(
        merged.get("evidence_chars_per_token", 4.0),
        "evidence_chars_per_token",
    )
    if evidence_chars_per_token <= 0:
        raise ConfigurationError("evidence_chars_per_token must be > 0")

    return FactLLaMAConfig(
        base_model=base_model,
        adapter_path=Path(adapter_path),
        load_in_4bit=_as_bool(merged.get("load_in_4bit", True), "load_in_4bit"),
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=_as_str(
            merged.get("bnb_4bit_compute_dtype", "bfloat16"),
            "bnb_4bit_compute_dtype",
        ),
        bnb_4bit_use_double_quant=_as_bool(
            merged.get("bnb_4bit_use_double_quant", True),
            "bnb_4bit_use_double_quant",
        ),
        device_map=_as_str(merged.get("device_map", "auto"), "device_map"),
        trust_remote_code=_as_bool(
            merged.get("trust_remote_code", False), "trust_remote_code"
        ),
        max_new_tokens=max_new_tokens,
        do_sample=_as_bool(merged.get("do_sample", False), "do_sample"),
        temperature=temperature,
        max_evidence_tokens=max_evidence_tokens,
        evidence_chars_per_token=evidence_chars_per_token,
    )


def load_config(
    path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> FactLLaMAConfig:
    """Load FactLLaMA configuration from JSON plus environment overrides.

    Resolution order (lowest to highest precedence):
        1. :data:`FactLLaMAConfig` defaults.
        2. ``configs/factllama.json`` (or ``path`` / ``FACTLLAMA_CONFIG``).
        3. ``FACTLLAMA_*`` environment variables.

    A missing default config file is fine; an explicitly requested file that
    does not exist is an error.
    """
    environ = os.environ if env is None else env

    requested = path if path is not None else environ.get(CONFIG_PATH_ENV)
    explicit = requested is not None and str(requested).strip() != ""

    if explicit:
        config_path = Path(str(requested)).expanduser()
        if not config_path.is_absolute():
            config_path = project_root() / config_path
        if not config_path.is_file():
            raise ConfigurationError(
                f"FactLLaMA config file not found: {config_path}"
            )
    else:
        config_path = project_root() / DEFAULT_CONFIG_PATH

    data: dict[str, Any] = {}
    if config_path.is_file():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(
                f"Could not read FactLLaMA config {config_path}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise ConfigurationError(
                f"FactLLaMA config {config_path} must contain a JSON object"
            )
        data = raw

    return _build_config(data, config_path, environ)


def _import_module(name: str) -> Any:
    """Import a dependency or raise a clear missing-dependency error."""
    try:
        return importlib.import_module(name)
    except (ImportError, OSError, RuntimeError, AttributeError) as exc:
        hint = _DEPENDENCY_HINTS.get(name, f"pip install {name}")
        raise MissingDependencyError(
            f"FactLLaMA inference requires '{name}' ({hint}). "
            f"Import failed: {exc}"
        ) from exc


def _looks_like_access_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _ACCESS_ERROR_MARKERS)


def _access_hint(target: str) -> str:
    return (
        f"Could not access {target}. For the gated Llama-3.2 checkpoint, "
        "accept the licence on huggingface.co, then run "
        "`huggingface-cli login` or set HF_TOKEN. For a local adapter, "
        "check FACTLLAMA_ADAPTER_PATH and file permissions."
    )


class FactLLaMAModelLoader:
    """Lazily load the quantized base model plus LoRA adapter.

    Args:
        config: Explicit configuration. When omitted, :func:`load_config`
            is called with ``config_path`` and ``env``.
        config_path: Optional JSON config path (see :func:`load_config`).
        env: Optional environment mapping for tests; defaults to
            ``os.environ``.
        modules: Optional pre-imported ``{"torch", "transformers", "peft",
            "accelerate", "bitsandbytes"}`` mapping. Used by tests so no
            heavy dependency is imported.

    The first :meth:`load` call validates the adapter, imports dependencies,
    checks 4-bit support, loads the tokenizer and base model, then attaches
    the adapter. Later calls return the cached bundle.
    """

    def __init__(
        self,
        config: FactLLaMAConfig | None = None,
        *,
        config_path: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        modules: Mapping[str, Any] | None = None,
    ) -> None:
        self._config = config or load_config(config_path, env)
        self._modules = dict(modules) if modules is not None else None
        self._loaded: LoadedFactLLaMA | None = None

    @property
    def config(self) -> FactLLaMAConfig:
        """The configuration this loader was created with."""
        return self._config

    @property
    def is_loaded(self) -> bool:
        """Whether the model bundle has already been loaded."""
        return self._loaded is not None

    def load(self) -> LoadedFactLLaMA:
        """Load and cache the model bundle."""
        if self._loaded is None:
            self._loaded = self._load()
        return self._loaded

    def _import_dependencies(self) -> dict[str, Any]:
        names = ["torch", "transformers", "peft", "accelerate"]
        if self._config.load_in_4bit:
            names.append("bitsandbytes")
        return {name: _import_module(name) for name in names}

    def _validate_environment(self, modules: Mapping[str, Any]) -> None:
        if not self._config.load_in_4bit:
            return

        if "bitsandbytes" not in modules:
            raise MissingDependencyError(
                "4-bit loading requires 'bitsandbytes' "
                f"({_DEPENDENCY_HINTS['bitsandbytes']})."
            )

        torch_module = modules["torch"]
        cuda = getattr(torch_module, "cuda", None)
        is_available = getattr(cuda, "is_available", None)
        if not callable(is_available) or not is_available():
            raise QuantizationUnsupportedError(
                "4-bit QLoRA inference requires an available CUDA GPU, but "
                "torch.cuda.is_available() returned False. Use a CUDA-capable "
                "machine, or set load_in_4bit=false / FACTLLAMA_LOAD_IN_4BIT=0 "
                "for a smaller non-quantized experiment (3B weights will not "
                "fit comfortably on CPU)."
            )

    def _load_tokenizer(
        self, modules: Mapping[str, Any], adapter_path: Path
    ) -> Any:
        transformers = modules["transformers"]
        source = (
            adapter_path
            if (adapter_path / TOKENIZER_CONFIG_FILENAME).is_file()
            else self._config.base_model
        )
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(str(source))
        except Exception as exc:  # noqa: BLE001 - boundary around the HF API
            if _looks_like_access_error(exc):
                raise ModelAccessError(_access_hint(str(source))) from exc
            raise ModelLoadError(
                f"Failed to load tokenizer from {source}: {exc}"
            ) from exc

        if (
            getattr(tokenizer, "pad_token", None) is None
            and getattr(tokenizer, "eos_token", None) is not None
        ):
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _torch_dtype(self, torch_module: Any) -> Any:
        name = self._config.bnb_4bit_compute_dtype
        dtype = getattr(torch_module, name, None)
        if dtype is None:
            raise ConfigurationError(
                "bnb_4bit_compute_dtype must be a torch dtype attribute "
                f"(e.g. bfloat16, float16, float32); got {name!r}"
            )
        return dtype

    def _load_base_model(self, modules: Mapping[str, Any]) -> Any:
        torch_module = modules["torch"]
        transformers = modules["transformers"]

        kwargs: dict[str, Any] = {
            "device_map": self._config.device_map,
            "trust_remote_code": self._config.trust_remote_code,
        }
        if self._config.load_in_4bit:
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self._config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=self._torch_dtype(torch_module),
                bnb_4bit_use_double_quant=(
                    self._config.bnb_4bit_use_double_quant
                ),
            )

        try:
            return transformers.AutoModelForCausalLM.from_pretrained(
                self._config.base_model, **kwargs
            )
        except Exception as exc:  # noqa: BLE001 - boundary around the HF API
            if _looks_like_access_error(exc):
                raise ModelAccessError(
                    _access_hint(self._config.base_model)
                ) from exc
            raise ModelLoadError(
                f"Failed to load base model {self._config.base_model}: {exc}"
            ) from exc

    def _attach_adapter(
        self,
        modules: Mapping[str, Any],
        model: Any,
        adapter_path: Path,
    ) -> Any:
        peft = modules["peft"]
        try:
            adapted = peft.PeftModel.from_pretrained(model, str(adapter_path))
        except Exception as exc:  # noqa: BLE001 - boundary around the peft API
            raise ModelLoadError(
                f"Failed to attach adapter from {adapter_path}: {exc}. "
                "Check that the folder matches the base model the adapter "
                "was trained on."
            ) from exc

        if hasattr(adapted, "eval"):
            adapted.eval()
        return adapted

    def _load(self) -> LoadedFactLLaMA:
        adapter_path = validate_adapter_path(self._config.adapter_path)
        modules = self._modules or self._import_dependencies()

        self._validate_environment(modules)
        tokenizer = self._load_tokenizer(modules, adapter_path)
        model = self._load_base_model(modules)
        model = self._attach_adapter(modules, model, adapter_path)

        return LoadedFactLLaMA(
            model=model,
            tokenizer=tokenizer,
            config=self._config,
            adapter_path=adapter_path,
        )
