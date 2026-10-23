"""FactLLaMA inference package.

Public entry points for evidence-based fact verification with the
fine-tuned Llama-3.2-3B-Instruct + QLoRA adapter.

Importing this package is lightweight: no ML dependency is imported and no
model weights are loaded. Model loading happens on the first
``FactLLaMAPredictor.predict`` call (or an explicit loader ``load`` call).
"""

from .model_loader import (
    AdapterNotFoundError,
    ConfigurationError,
    FactLLaMAConfig,
    FactLLaMAError,
    FactLLaMAModelLoader,
    LoadedFactLLaMA,
    MissingDependencyError,
    ModelAccessError,
    ModelLoadError,
    QuantizationUnsupportedError,
    load_config,
    resolve_adapter_path,
    validate_adapter_path,
)
from .output_parser import (
    VERDICT_LABELS,
    VerdictParseError,
    parse_verdict,
    try_parse_verdict,
)
from .predictor import (
    FactLLaMAPredictor,
    LlamaTextGenerator,
    Prediction,
    PredictionError,
)
from .prompt_builder import (
    SYSTEM_PROMPT,
    PromptBuildError,
    build_messages,
    build_user_prompt,
    normalise_evidence,
)

__all__ = [
    "AdapterNotFoundError",
    "ConfigurationError",
    "FactLLaMAConfig",
    "FactLLaMAError",
    "FactLLaMAModelLoader",
    "FactLLaMAPredictor",
    "LlamaTextGenerator",
    "LoadedFactLLaMA",
    "MissingDependencyError",
    "ModelAccessError",
    "ModelLoadError",
    "Prediction",
    "PredictionError",
    "PromptBuildError",
    "QuantizationUnsupportedError",
    "SYSTEM_PROMPT",
    "VERDICT_LABELS",
    "VerdictParseError",
    "build_messages",
    "build_user_prompt",
    "load_config",
    "normalise_evidence",
    "parse_verdict",
    "resolve_adapter_path",
    "try_parse_verdict",
    "validate_adapter_path",
]
