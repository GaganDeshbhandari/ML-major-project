"""FactLLaMA prediction entry point.

``FactLLaMAPredictor`` ties prompt construction, generation and verdict
parsing together. The model is loaded lazily on the first :meth:`predict`
call, so importing this module (or running unit tests) never loads weights.

For tests and future integrations, both the model loader and the text
generator can be injected:

    predictor = FactLLaMAPredictor(loader=my_loader)
    predictor = FactLLaMAPredictor(generator=my_generator)

If no generator is injected, a :class:`LlamaTextGenerator` is built from the
loaded model/tokenizer. If no loader is injected, a
:class:`~src.factllama.model_loader.FactLLaMAModelLoader` is created from the
project configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence

from .model_loader import (
    FactLLaMAConfig,
    FactLLaMAModelLoader,
    FactLLaMAError,
    MissingDependencyError,
)
from .output_parser import try_parse_verdict
from .prompt_builder import (
    DEFAULT_EVIDENCE_BUDGET,
    EvidenceBudget,
    build_prompt,
    render_chat_fallback,
)

__all__ = [
    "FactLLaMAPredictor",
    "LlamaTextGenerator",
    "Prediction",
    "PredictionError",
    "TextGenerator",
]


class PredictionError(FactLLaMAError):
    """Raised when tokenization or generation cannot produce usable text."""


@dataclass(frozen=True)
class Prediction:
    """One verification result.

    ``verdict`` is ``None`` when the model output could not be parsed; the
    raw generation is always kept for inspection. ``evidence_used`` is the
    approximate evidence-token count, not a passage count; the other budget
    fields report whether the prompt was reduced before generation.
    """

    claim: str
    verdict: str | None
    raw_output: str
    evidence_used: int = 0
    evidence_excluded: int = 0
    evidence_truncated: bool = False


class TextGenerator(Protocol):
    """Minimal generator interface used by the predictor."""

    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        """Return generated text for chat messages."""


class LlamaTextGenerator:
    """Generate text with a loaded causal LM and tokenizer.

    Args:
        model: Loaded model (base + adapter).
        tokenizer: Matching tokenizer.
        max_new_tokens: Generation budget. Verdicts are short, so a small
            value also limits the chance of explanations after the label.
        do_sample: Sampling toggle (default deterministic).
        temperature: Sampling temperature; ignored when ``do_sample`` is
            false.
        torch_module: Injected torch module for tests; imported lazily
            otherwise.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_new_tokens: int = 16,
        do_sample: bool = False,
        temperature: float = 0.0,
        torch_module: Any = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._max_new_tokens = max_new_tokens
        self._do_sample = do_sample
        self._temperature = temperature
        self._torch_module = torch_module

    def _torch(self) -> Any:
        if self._torch_module is None:
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise MissingDependencyError(
                    "FactLLaMA generation requires 'torch'. "
                    "Install it with pip install 'torch>=2.2'."
                ) from exc
            self._torch_module = torch
        return self._torch_module

    def _render_prompt(self, messages: Sequence[dict[str, str]]) -> str:
        if not messages:
            raise PredictionError("messages must not be empty")

        if getattr(self._tokenizer, "chat_template", None):
            return self._tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
            )
        return render_chat_fallback(messages)

    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        """Tokenize chat messages, generate, and decode only new tokens."""
        prompt = self._render_prompt(messages)
        encoded = self._tokenizer(prompt, return_tensors="pt")

        try:
            input_ids = encoded["input_ids"]
        except (KeyError, TypeError) as exc:
            raise PredictionError(
                "tokenizer did not return input_ids"
            ) from exc

        prompt_length = int(input_ids.shape[-1])
        device = getattr(self._model, "device", None)
        model_inputs = {
            key: value.to(device) if device is not None and hasattr(value, "to") else value
            for key, value in encoded.items()
        }

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": self._do_sample,
        }
        if self._do_sample:
            generation_kwargs["temperature"] = self._temperature

        pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self._tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id

        torch_module = self._torch()
        with torch_module.inference_mode():
            output_ids = self._model.generate(
                **model_inputs, **generation_kwargs
            )

        new_tokens = output_ids[0][prompt_length:]
        return self._tokenizer.decode(
            new_tokens, skip_special_tokens=True
        )


class FactLLaMAPredictor:
    """Run evidence-based fact verification with the fine-tuned adapter.

    Args:
        loader: Injected loader. When omitted, a default
            :class:`FactLLaMAModelLoader` is created from ``config`` (or the
            project config file) on first use.
        config: Explicit configuration used only when creating the default
            loader.
        generator: Injected text generator; skips model loading entirely.
        evidence_budget: Optional evidence budget override. When omitted, the
            loader/config values are used, falling back to
            :data:`~src.factllama.prompt_builder.DEFAULT_EVIDENCE_BUDGET`.

    Raises:
        ValueError: both ``loader`` and ``config`` were supplied, which is
            ambiguous.
    """

    def __init__(
        self,
        loader: FactLLaMAModelLoader | None = None,
        *,
        config: FactLLaMAConfig | None = None,
        generator: TextGenerator | None = None,
        evidence_budget: EvidenceBudget | None = None,
    ) -> None:
        if loader is not None and config is not None:
            raise ValueError(
                "Pass either an explicit loader or a config, not both"
            )
        self._loader = loader
        self._config = config
        self._generator = generator
        self._evidence_budget = evidence_budget

    def _resolve_evidence_budget(self) -> EvidenceBudget:
        """Use the explicit budget, then config values, then the default."""
        if self._evidence_budget is not None:
            return self._evidence_budget

        config = self._config
        if config is None and self._loader is not None:
            config = getattr(self._loader, "config", None)

        if isinstance(config, FactLLaMAConfig):
            return EvidenceBudget(
                max_tokens=config.max_evidence_tokens,
                chars_per_token=config.evidence_chars_per_token,
            )
        return DEFAULT_EVIDENCE_BUDGET

    def _ensure_generator(self) -> TextGenerator:
        if self._generator is None:
            if self._loader is None:
                self._loader = FactLLaMAModelLoader(self._config)
            loaded = self._loader.load()
            self._generator = LlamaTextGenerator(
                loaded.model,
                loaded.tokenizer,
                max_new_tokens=loaded.config.max_new_tokens,
                do_sample=loaded.config.do_sample,
                temperature=loaded.config.temperature,
            )
        return self._generator

    def predict(self, claim: str, evidence: Any) -> Prediction:
        """Verify one claim against its evidence.

        Prompt/input errors raise
        :class:`~src.factllama.prompt_builder.PromptBuildError`. Model output
        that contains no supported verdict yields ``Prediction.verdict is
        None`` rather than an exception. Truncation/exclusion of evidence is
        reported on the returned prediction.
        """
        built = build_prompt(
            claim,
            evidence,
            budget=self._resolve_evidence_budget(),
        )
        raw_output = self._ensure_generator().generate(built.messages)
        return Prediction(
            claim=claim.strip(),
            verdict=try_parse_verdict(raw_output),
            raw_output=raw_output,
            evidence_used=built.selection.estimated_tokens,
            evidence_excluded=built.selection.excluded,
            evidence_truncated=built.selection.truncated,
        )

    def predict_batch(
        self,
        items: Iterable[tuple[str, Any]],
    ) -> list[Prediction]:
        """Verify ``(claim, evidence)`` pairs sequentially."""
        return [self.predict(claim, evidence) for claim, evidence in items]
