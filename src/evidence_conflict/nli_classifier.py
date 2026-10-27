"""NLI stance classification for evidence conflict scoring.

The classifier compares each evidence passage with a claim using a
three-class natural-language-inference model.  Input pairs are always
``(evidence, claim)``: evidence is the NLI premise and the claim is the NLI
hypothesis.  This is the reverse of the retrieval reranker's
``(claim, evidence)`` pair order.

Importing this module does not import PyTorch or Transformers, probe devices,
download weights, or load a model.  The real backend is imported lazily and
loaded on the first non-empty :meth:`NLIStanceClassifier.predict` call.

Confidence is the model's maximum class probability.  It is not a calibrated
probability that the predicted stance is correct.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any

from .config import ECSConfig
from .errors import (
    StanceInferenceError,
    StanceInputError,
    StanceModelError,
    UnsupportedLabelMappingError,
)
from .schemas import StanceLabel, StancePrediction, StanceProbabilities

__all__ = ["NLIStanceClassifier"]

_STANCE_ORDER: tuple[StanceLabel, ...] = (
    StanceLabel.SUPPORTS,
    StanceLabel.REFUTES,
    StanceLabel.NOT_ENOUGH_INFO,
)

#: Semantic aliases accepted from model label metadata.  Aliases are
#: deliberately disjoint and exclude generic names such as ``LABEL_0``.
_LABEL_ALIASES: Mapping[StanceLabel, frozenset[str]] = MappingProxyType(
    {
        StanceLabel.SUPPORTS: frozenset(
            {
                "supports",
                "support",
                "entailment",
                "entails",
                "entailed",
                "entail",
            }
        ),
        StanceLabel.REFUTES: frozenset(
            {
                "refutes",
                "refute",
                "refutation",
                "contradiction",
                "contradicts",
                "contradict",
                "contradictory",
            }
        ),
        StanceLabel.NOT_ENOUGH_INFO: frozenset(
            {
                "not_enough_info",
                "not_enough_information",
                "neutral",
                "neither",
                "unknown",
                "insufficient",
                "nei",
            }
        ),
    }
)

_EXPECTED_LABEL_INDICES = frozenset({0, 1, 2})


@dataclass(slots=True)
class _EvidencePassage:
    """Normalized evidence text and its original position."""

    index: int
    text: str
    truncated: bool


def _normalize_claim(claim: object) -> str:
    """Return a stripped, non-empty claim or raise ``StanceInputError``."""

    if not isinstance(claim, str):
        raise StanceInputError(
            "claim must be a non-empty string; "
            f"got {type(claim).__name__}."
        )
    text = claim.strip()
    if not text:
        raise StanceInputError("claim must be a non-empty string.")
    return text


def _evidence_text(item: object, index: int) -> str:
    """Extract evidence text from a plain string or a mapping record."""

    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        if "content" not in item:
            raise StanceInputError(
                f"evidence[{index}] mapping must contain a 'content' string."
            )
        content = item["content"]
        if not isinstance(content, str):
            raise StanceInputError(
                f"evidence[{index}]['content'] must be a string; "
                f"got {type(content).__name__}."
            )
        return content
    raise StanceInputError(
        f"evidence[{index}] must be a string or a mapping with a "
        f"'content' string; got {type(item).__name__}."
    )


def _normalize_evidence(
    evidence: Iterable[str | Mapping[str, Any]],
    max_chars: int,
) -> list[_EvidencePassage]:
    """Normalize evidence without mutating caller-owned records."""

    if isinstance(evidence, (str, bytes, bytearray, Mapping)):
        raise StanceInputError(
            "evidence must be an iterable of passages, not a single "
            "string or mapping."
        )
    try:
        items = list(evidence)
    except TypeError as exc:
        raise StanceInputError(
            "evidence must be an iterable of passages."
        ) from exc

    passages: list[_EvidencePassage] = []
    for index, item in enumerate(items):
        text = _evidence_text(item, index).strip()
        if not text:
            raise StanceInputError(
                f"evidence[{index}] must contain non-empty text."
            )
        truncated = False
        if len(text) > max_chars:
            text = text[:max_chars]
            truncated = True
        passages.append(
            _EvidencePassage(
                index=index,
                text=text,
                truncated=truncated,
            )
        )
    return passages


def _normalize_label_name(label: str) -> str:
    """Normalize a model label for semantic alias matching."""

    return "_".join(
        label.strip().casefold().replace("-", " ").split()
    )


def _coerce_label_index(value: object, *, field: str) -> int:
    """Return an integer label index from an id2label key or label2id value."""

    if isinstance(value, bool):
        raise UnsupportedLabelMappingError(
            f"{field} must be an integer index; got a boolean."
        )
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isascii() and stripped.isdigit():
            return int(stripped)
    raise UnsupportedLabelMappingError(
        f"{field} must be an integer index; got {value!r}."
    )


def _parse_id2label(id2label: Mapping[object, object]) -> dict[int, str]:
    """Convert ``id2label`` metadata into a validated index-to-label map."""

    parsed: dict[int, str] = {}
    for raw_index, raw_label in id2label.items():
        index = _coerce_label_index(
            raw_index,
            field="id2label key",
        )
        if index in parsed:
            raise UnsupportedLabelMappingError(
                f"Duplicate id2label index {index}."
            )
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise UnsupportedLabelMappingError(
                f"id2label[{index}] must be a non-empty string."
            )
        parsed[index] = raw_label.strip()
    return parsed


def _invert_label2id(label2id: Mapping[object, object]) -> dict[int, str]:
    """Invert ``label2id`` metadata into an index-to-label map."""

    parsed: dict[int, str] = {}
    for raw_label, raw_index in label2id.items():
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise UnsupportedLabelMappingError(
                "label2id keys must be non-empty label strings."
            )
        index = _coerce_label_index(
            raw_index,
            field=f"label2id[{raw_label!r}]",
        )
        if index in parsed:
            raise UnsupportedLabelMappingError(
                f"Duplicate label2id index {index}."
            )
        parsed[index] = raw_label.strip()
    return parsed


def _mapping_from_index_labels(
    index_labels: Mapping[int, str],
) -> dict[StanceLabel, int]:
    """Map a three-class index-to-label map onto application stance labels."""

    if (
        len(index_labels) != 3
        or set(index_labels) != _EXPECTED_LABEL_INDICES
    ):
        raise UnsupportedLabelMappingError(
            "NLI model must expose exactly three contiguous labels at "
            "indices 0, 1, and 2; got indices "
            f"{sorted(index_labels)}."
        )

    result: dict[StanceLabel, int] = {}
    for index in range(3):
        raw_label = index_labels[index]
        normalized = _normalize_label_name(raw_label)
        matches = [
            label
            for label, aliases in _LABEL_ALIASES.items()
            if normalized in aliases
        ]
        if not matches:
            raise UnsupportedLabelMappingError(
                f"Unsupported or non-semantic NLI label {raw_label!r} at "
                f"index {index}. Generic labels such as LABEL_0 are not "
                "guessed."
            )
        if len(matches) > 1:
            raise UnsupportedLabelMappingError(
                f"Ambiguous NLI label {raw_label!r} at index {index}."
            )
        label = matches[0]
        if label in result:
            raise UnsupportedLabelMappingError(
                f"Duplicate NLI label mapping: {raw_label!r} at index "
                f"{index} maps to {label.value!r}, which is already mapped."
            )
        result[label] = index

    if set(result) != set(StanceLabel):
        missing = [
            label.value
            for label in _STANCE_ORDER
            if label not in result
        ]
        raise UnsupportedLabelMappingError(
            "NLI label mapping is missing application labels: "
            + ", ".join(missing)
        )
    return result


def _resolve_label_indices(model: Any) -> dict[StanceLabel, int]:
    """Resolve a model's semantic label metadata without assuming order."""

    config = getattr(model, "config", None)
    if config is None:
        raise UnsupportedLabelMappingError(
            "NLI model has no configuration; cannot determine its labels."
        )

    id2label = getattr(config, "id2label", None)
    label2id = getattr(config, "label2id", None)

    if isinstance(id2label, Mapping) and id2label:
        index_labels = _parse_id2label(id2label)
    elif isinstance(label2id, Mapping) and label2id:
        index_labels = _invert_label2id(label2id)
    else:
        raise UnsupportedLabelMappingError(
            "NLI model configuration has no id2label or label2id mapping."
        )

    return _mapping_from_index_labels(index_labels)


def _validate_label_indices(
    mapping: Mapping[StanceLabel, int],
) -> dict[StanceLabel, int]:
    """Validate an injected label-to-column mapping."""

    if not isinstance(mapping, Mapping):
        raise UnsupportedLabelMappingError(
            "label_indices must be a mapping from StanceLabel to column "
            "index."
        )

    result: dict[StanceLabel, int] = {}
    used: set[int] = set()
    for label, index in mapping.items():
        if not isinstance(label, StanceLabel):
            raise UnsupportedLabelMappingError(
                "label_indices keys must be StanceLabel values; "
                f"got {label!r}."
            )
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise UnsupportedLabelMappingError(
                f"label_indices[{label.value!r}] must be an integer."
            )
        normalized_index = int(index)
        if normalized_index not in _EXPECTED_LABEL_INDICES:
            raise UnsupportedLabelMappingError(
                f"label_indices[{label.value!r}] must be 0, 1, or 2; "
                f"got {normalized_index}."
            )
        if normalized_index in used:
            raise UnsupportedLabelMappingError(
                "label_indices maps multiple labels to column "
                f"{normalized_index}."
            )
        used.add(normalized_index)
        result[label] = normalized_index

    if set(result) != set(StanceLabel):
        raise UnsupportedLabelMappingError(
            "label_indices must define exactly supports, refutes, and "
            "not_enough_info."
        )
    return result


def _resolve_device(torch_module: Any, requested: str) -> str:
    """Resolve ``auto`` and validate explicitly requested CUDA devices."""

    if requested == "cpu":
        return "cpu"

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        if requested == "auto":
            return "cpu"
        raise StanceModelError(
            f"Device {requested!r} was requested, but torch.cuda is "
            "unavailable."
        )

    try:
        available = bool(cuda.is_available())
    except Exception as exc:
        raise StanceModelError(
            "Could not query CUDA availability from PyTorch."
        ) from exc

    if requested == "auto":
        return "cuda" if available else "cpu"

    if not available:
        raise StanceModelError(
            f"Device {requested!r} was requested, but CUDA is not available."
        )

    if requested.startswith("cuda:"):
        index_text = requested.split(":", 1)[1]
        try:
            index = int(index_text)
        except ValueError as exc:
            raise StanceModelError(
                f"Invalid CUDA device {requested!r}."
            ) from exc

        device_count = getattr(cuda, "device_count", None)
        if callable(device_count):
            try:
                count = int(device_count())
            except Exception as exc:
                raise StanceModelError(
                    "Could not query the number of CUDA devices."
                ) from exc
            if index >= count:
                raise StanceModelError(
                    f"CUDA device index {index} is out of range; "
                    f"PyTorch reports {count} device(s)."
                )
    elif requested != "cuda":
        raise StanceModelError(
            f"Unsupported device {requested!r}; expected auto, cpu, cuda, "
            "or cuda:<index>."
        )

    return requested


def _extract_logits(outputs: Any) -> Any:
    """Extract logits from a Hugging Face model output or a fake result."""

    if isinstance(outputs, Mapping):
        logits = outputs.get("logits")
    else:
        logits = getattr(outputs, "logits", None)

    if logits is None and isinstance(outputs, (tuple, list)) and outputs:
        logits = outputs[0]
    if logits is None and hasattr(outputs, "tolist"):
        logits = outputs
    if logits is None:
        raise StanceInferenceError(
            "NLI model output does not contain logits."
        )
    return logits


def _coerce_logits(raw: Any, expected_rows: int) -> list[list[float]]:
    """Validate and convert raw logits to a finite 2-D Python list."""

    if hasattr(raw, "detach"):
        raw = raw.detach()
    if hasattr(raw, "cpu"):
        raw = raw.cpu()
    if hasattr(raw, "tolist"):
        raw = raw.tolist()

    if not isinstance(raw, (list, tuple)):
        raise StanceInferenceError(
            "NLI model returned logits of type "
            f"{type(raw).__name__}; expected a 2-D array."
        )
    if len(raw) != expected_rows:
        raise StanceInferenceError(
            f"NLI model returned {len(raw)} logit rows for "
            f"{expected_rows} evidence passages."
        )

    rows: list[list[float]] = []
    for row_index, row in enumerate(raw):
        if hasattr(row, "detach"):
            row = row.detach()
        if hasattr(row, "cpu"):
            row = row.cpu()
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, (list, tuple)):
            raise StanceInferenceError(
                f"NLI logits row {row_index} must be a sequence; "
                f"got {type(row).__name__}."
            )
        if len(row) != 3:
            raise StanceInferenceError(
                f"NLI logits row {row_index} has {len(row)} classes; "
                "expected exactly 3."
            )

        values: list[float] = []
        for column, value in enumerate(row):
            if isinstance(value, bool) or isinstance(value, (str, bytes)):
                raise StanceInferenceError(
                    "NLI logit at row "
                    f"{row_index}, column {column} is not a real number: "
                    f"{value!r}."
                )
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise StanceInferenceError(
                    "NLI logit at row "
                    f"{row_index}, column {column} is not a real number: "
                    f"{value!r}."
                ) from exc
            if not math.isfinite(number):
                raise StanceInferenceError(
                    "NLI logit at row "
                    f"{row_index}, column {column} is not finite: "
                    f"{number!r}."
                )
            values.append(number)
        rows.append(values)

    return rows


def _softmax_probabilities(logits: Sequence[float]) -> list[float]:
    """Apply a numerically stable softmax to one three-class logit row."""

    maximum = max(logits)
    exponentials = [
        math.exp(value - maximum)
        for value in logits
    ]
    total = sum(exponentials)
    if not math.isfinite(total) or total <= 0.0:
        raise StanceInferenceError(
            "NLI softmax normalization produced an invalid total."
        )

    probabilities = [
        value / total
        for value in exponentials
    ]
    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in probabilities
    ):
        raise StanceInferenceError(
            "NLI softmax produced invalid class probabilities."
        )
    return probabilities


def _move_to_device(encoded: Any, device: str) -> Any:
    """Move tokenized tensors to ``device`` without assuming a backend."""

    if not isinstance(encoded, Mapping):
        return encoded

    moved: dict[Any, Any] = {}
    for key, value in encoded.items():
        to_method = getattr(value, "to", None)
        moved[key] = to_method(device) if callable(to_method) else value
    return moved


class NLIStanceClassifier:
    """Three-class NLI stance classifier for claim-evidence pairs.

    Args:
        config: Evidence conflict configuration.
        _tokenizer: Injected tokenizer for tests. When omitted, the real
            tokenizer is loaded lazily.
        _model: Injected model for tests or a preloaded model. When omitted,
            the real model is loaded lazily.
        _predict_fn: Injected inference callable for tests. It receives the
            tokenized batch and returns logits shaped ``(batch, 3)``. When
            set, no model is loaded for inference.
        _torch: Injected PyTorch module for tests.
        _label_indices: Injected label-to-column mapping for tests. When
            omitted, it is resolved from the model configuration.

    Model loading, device selection, and label mapping happen on the first
    non-empty :meth:`predict` call.  An empty evidence sequence returns
    immediately without loading the tokenizer or model.
    """

    def __init__(
        self,
        config: ECSConfig,
        *,
        _tokenizer: Any | None = None,
        _model: Any | None = None,
        _predict_fn: Callable[[Any], Any] | None = None,
        _torch: Any | None = None,
        _label_indices: Mapping[StanceLabel, int] | None = None,
    ) -> None:
        if not isinstance(config, ECSConfig):
            raise TypeError("config must be an ECSConfig instance.")

        self.config = config
        self._tokenizer = _tokenizer
        self._model = _model
        self._predict_fn = _predict_fn
        self._torch = _torch
        self._label_indices = (
            _validate_label_indices(_label_indices)
            if _label_indices is not None
            else None
        )
        self._device: str | None = None
        self._model_prepared = False

    def predict(
        self,
        claim: str,
        evidence: Iterable[str | Mapping[str, Any]],
    ) -> list[StancePrediction]:
        """Return one stance prediction per evidence passage.

        Evidence may be plain text or a mapping with a ``content`` string.
        The original evidence records are never modified.  Predictions are
        returned in input order and use the original evidence indices.
        """

        claim_text = _normalize_claim(claim)
        passages = _normalize_evidence(
            evidence,
            self.config.max_evidence_chars,
        )
        if not passages:
            return []

        tokenizer = self._get_tokenizer()
        self._validate_claim_fits(tokenizer, claim_text)
        self._mark_token_truncation(tokenizer, claim_text, passages)
        self._prepare_inference()

        predictions: list[StancePrediction] = []
        for start in range(0, len(passages), self.config.batch_size):
            batch = passages[start : start + self.config.batch_size]
            predictions.extend(
                self._predict_batch(tokenizer, claim_text, batch)
            )
        return predictions

    def classify(
        self,
        claim: str,
        evidence: Iterable[str | Mapping[str, Any]],
    ) -> list[StancePrediction]:
        """Alias for :meth:`predict`."""

        return self.predict(claim, evidence)

    def _get_tokenizer(self) -> Any:
        """Load and cache the tokenizer on first use."""

        if self._tokenizer is not None:
            self._check_tokenizer_max_length(self._tokenizer)
            return self._tokenizer

        try:
            from transformers import AutoTokenizer
        except Exception as exc:
            raise StanceModelError(
                "NLI tokenizer backend unavailable: transformers could not "
                "be imported. Install torch and transformers to use the "
                "stance classifier."
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                revision=self.config.model_revision,
                local_files_only=self.config.local_files_only,
            )
        except Exception as exc:
            raise StanceModelError(
                "Failed to load NLI tokenizer "
                f"{self.config.model_name!r}: {exc}"
            ) from exc

        self._check_tokenizer_max_length(tokenizer)
        self._tokenizer = tokenizer
        return tokenizer

    def _check_tokenizer_max_length(self, tokenizer: Any) -> None:
        """Reject a configured length beyond the tokenizer's model limit."""

        model_max_length = getattr(tokenizer, "model_max_length", None)
        if (
            isinstance(model_max_length, Integral)
            and not isinstance(model_max_length, bool)
            and 0 < int(model_max_length) < self.config.max_length
        ):
            raise StanceModelError(
                f"Configured max_length={self.config.max_length} exceeds "
                f"tokenizer model_max_length={int(model_max_length)} for "
                f"{self.config.model_name!r}."
            )

    def _get_torch(self) -> Any:
        """Import and cache PyTorch on first use."""

        if self._torch is not None:
            return self._torch

        try:
            import torch
        except Exception as exc:
            raise StanceModelError(
                "NLI inference backend unavailable: PyTorch could not be "
                "imported. Install torch and transformers to use the "
                "stance classifier."
            ) from exc

        self._torch = torch
        return torch

    def _get_model(self) -> Any:
        """Load and cache the sequence-classification model on first use."""

        if self._model is not None:
            return self._model

        try:
            from transformers import AutoModelForSequenceClassification
        except Exception as exc:
            raise StanceModelError(
                "NLI model backend unavailable: transformers could not be "
                "imported. Install torch and transformers to use the "
                "stance classifier."
            ) from exc

        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                revision=self.config.model_revision,
                local_files_only=self.config.local_files_only,
            )
        except Exception as exc:
            raise StanceModelError(
                "Failed to load NLI model "
                f"{self.config.model_name!r}: {exc}"
            ) from exc

        self._model = model
        return model

    def _prepare_inference(self) -> None:
        """Resolve label metadata, device, and model state once."""

        if self._predict_fn is not None:
            if self._label_indices is None:
                if self._model is None:
                    raise StanceModelError(
                        "Injected _predict_fn requires _label_indices or a "
                        "model with label metadata."
                    )
                self._label_indices = _resolve_label_indices(self._model)
            return

        torch_module = self._get_torch()
        if self._device is None:
            self._device = _resolve_device(
                torch_module,
                self.config.device,
            )

        model = self._get_model()
        if not self._model_prepared:
            try:
                model.eval()
            except Exception as exc:
                raise StanceModelError(
                    "Failed to put the NLI model in evaluation mode: "
                    f"{exc}"
                ) from exc
            try:
                model.to(self._device)
            except Exception as exc:
                raise StanceModelError(
                    "Failed to move the NLI model to device "
                    f"{self._device!r}: {exc}"
                ) from exc
            self._model_prepared = True

        if self._label_indices is None:
            self._label_indices = _resolve_label_indices(model)

    def _tokenize_single_pair(
        self,
        tokenizer: Any,
        first: str,
        second: str,
        *,
        context: str,
    ) -> Any:
        """Tokenize one pair without truncating either side."""

        try:
            return tokenizer(
                first,
                second,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                verbose=False,
            )
        except Exception as exc:
            raise StanceInferenceError(
                f"NLI tokenization failed during {context}: {exc}"
            ) from exc

    def _validate_claim_fits(self, tokenizer: Any, claim: str) -> None:
        """Ensure the claim plus special tokens leaves room for evidence."""

        encoded = self._tokenize_single_pair(
            tokenizer,
            "",
            claim,
            context="claim validation",
        )
        try:
            claim_length = len(encoded["input_ids"])
        except (KeyError, TypeError) as exc:
            raise StanceInferenceError(
                "NLI tokenizer did not return input_ids while validating "
                "the claim."
            ) from exc

        if claim_length >= self.config.max_length:
            raise StanceInputError(
                f"Claim requires {claim_length} tokens including special "
                f"tokens, leaving no room for evidence under "
                f"max_length={self.config.max_length}. The claim is never "
                "truncated."
            )

    def _mark_token_truncation(
        self,
        tokenizer: Any,
        claim: str,
        passages: list[_EvidencePassage],
    ) -> None:
        """Record token-level evidence truncation before batched inference."""

        for passage in passages:
            encoded = self._tokenize_single_pair(
                tokenizer,
                passage.text,
                claim,
                context=f"evidence[{passage.index}] truncation check",
            )
            try:
                length = len(encoded["input_ids"])
            except (KeyError, TypeError) as exc:
                raise StanceInferenceError(
                    "NLI tokenizer did not return input_ids while checking "
                    "evidence length."
                ) from exc
            if length > self.config.max_length:
                passage.truncated = True

    def _predict_batch(
        self,
        tokenizer: Any,
        claim: str,
        batch: list[_EvidencePassage],
    ) -> list[StancePrediction]:
        """Run inference for one batch of normalized evidence passages."""

        # Evidence is the first sequence (the NLI premise), so ``only_first``
        # truncates evidence rather than the claim (the second sequence).
        try:
            encoded = tokenizer(
                [passage.text for passage in batch],
                [claim] * len(batch),
                padding=True,
                truncation="only_first",
                max_length=self.config.max_length,
                return_tensors="pt",
            )
        except Exception as exc:
            raise StanceInferenceError(
                f"NLI tokenization failed: {exc}"
            ) from exc

        if self._predict_fn is not None:
            try:
                raw_logits = self._predict_fn(encoded)
            except StanceInferenceError:
                raise
            except Exception as exc:
                raise StanceInferenceError(
                    f"NLI inference failed: {exc}"
                ) from exc
        else:
            encoded = _move_to_device(encoded, self._device)
            try:
                with self._torch.inference_mode():
                    outputs = self._model(**encoded)
            except StanceInferenceError:
                raise
            except Exception as exc:
                raise StanceInferenceError(
                    f"NLI inference failed: {exc}"
                ) from exc
            raw_logits = _extract_logits(outputs)

        logits = _coerce_logits(raw_logits, len(batch))
        return [
            self._build_prediction(passage, row)
            for passage, row in zip(batch, logits)
        ]

    def _build_prediction(
        self,
        passage: _EvidencePassage,
        logits: list[float],
    ) -> StancePrediction:
        """Convert one logit row into a validated stance prediction."""

        probabilities = _softmax_probabilities(logits)
        label_indices = self._label_indices
        if label_indices is None:
            raise StanceInferenceError(
                "NLI label mapping was not initialized."
            )

        by_label = {
            label: probabilities[label_indices[label]]
            for label in _STANCE_ORDER
        }
        confidence = max(by_label.values())
        label = max(
            _STANCE_ORDER,
            key=lambda item: by_label[item],
        )

        threshold = self.config.confidence_threshold
        abstained = threshold is not None and confidence < threshold
        if abstained:
            label = StanceLabel.NOT_ENOUGH_INFO

        try:
            stance_probabilities = StanceProbabilities(
                supports=by_label[StanceLabel.SUPPORTS],
                refutes=by_label[StanceLabel.REFUTES],
                not_enough_info=by_label[
                    StanceLabel.NOT_ENOUGH_INFO
                ],
            )
            return StancePrediction(
                evidence_index=passage.index,
                label=label,
                confidence=confidence,
                probabilities=stance_probabilities,
                truncated=passage.truncated,
                abstained=abstained,
            )
        except StanceInputError as exc:
            raise StanceInferenceError(
                "NLI model produced invalid stance probabilities: "
                f"{exc}"
            ) from exc
