"""Unit tests for the NLI stance classifier.

No model downloads, network access, or GPU are used.  Real PyTorch and
Transformers imports are replaced with fakes or bypassed with an injected
prediction function.
"""

from __future__ import annotations

import math
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.evidence_conflict.config import ECSConfig
from src.evidence_conflict.errors import (
    StanceInferenceError,
    StanceInputError,
    StanceModelError,
    UnsupportedLabelMappingError,
)
from src.evidence_conflict.nli_classifier import NLIStanceClassifier
from src.evidence_conflict.schemas import StanceLabel, StancePrediction

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ID2LABEL = {
    0: "contradiction",
    1: "entailment",
    2: "neutral",
}
DEFAULT_LABEL_INDICES = {
    StanceLabel.SUPPORTS: 1,
    StanceLabel.REFUTES: 0,
    StanceLabel.NOT_ENOUGH_INFO: 2,
}


class FakeTokenizer:
    """Small tokenizer fake that models pair lengths and records calls."""

    def __init__(
        self,
        *,
        special_tokens: int = 3,
        model_max_length: int | None = None,
    ) -> None:
        self.special_tokens = special_tokens
        self.model_max_length = model_max_length
        self.calls: list[tuple[object, object, dict[str, object]]] = []

    def __call__(
        self,
        first: object,
        second: object | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        self.calls.append((first, second, kwargs))

        if isinstance(first, list):
            claims = (
                second
                if isinstance(second, list)
                else [second] * len(first)
            )
            input_ids: list[list[int]] = []
            for evidence, claim in zip(first, claims):
                length = self._pair_length(evidence, claim)
                max_length = kwargs.get("max_length")
                if (
                    kwargs.get("truncation") == "only_first"
                    and isinstance(max_length, int)
                ):
                    length = min(length, max_length)
                input_ids.append([0] * length)
            return {
                "input_ids": input_ids,
                "attention_mask": [
                    [1] * len(ids)
                    for ids in input_ids
                ],
            }

        if second is None:
            length = self._count(first)
            if kwargs.get("add_special_tokens") is not False:
                length += self.special_tokens
            return {"input_ids": [0] * length}

        return {
            "input_ids": [0] * self._pair_length(first, second),
        }

    def _count(self, text: object) -> int:
        return len(str(text).split())

    def _pair_length(self, first: object, second: object) -> int:
        return (
            self._count(first)
            + self._count(second)
            + self.special_tokens
        )


class FakeModel:
    """Minimal sequence-classification model fake."""

    def __init__(
        self,
        *,
        id2label: Mapping[object, object] | None = None,
        logits: object = None,
        fail: Exception | None = None,
    ) -> None:
        labels = DEFAULT_ID2LABEL if id2label is None else id2label
        self.config = SimpleNamespace(
            id2label=labels,
            label2id={
                label: index
                for index, label in labels.items()
            },
        )
        self.logits = logits
        self.fail = fail
        self.eval_calls = 0
        self.to_calls: list[str] = []
        self.calls: list[dict[str, object]] = []

    def eval(self) -> FakeModel:
        self.eval_calls += 1
        return self

    def to(self, device: str) -> FakeModel:
        self.to_calls.append(device)
        return self

    def __call__(self, **encoded: object) -> object:
        self.calls.append(encoded)
        if self.fail is not None:
            raise self.fail

        batch_size = len(encoded["input_ids"])
        if callable(self.logits):
            logits = self.logits(batch_size)
        elif self.logits is not None:
            logits = self.logits
        else:
            logits = [[0.0, 1.0, 2.0]] * batch_size
        return SimpleNamespace(logits=logits)


class _InferenceMode:
    def __enter__(self) -> _InferenceMode:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class FakeCuda:
    def __init__(self, available: bool, count: int) -> None:
        self.available = available
        self.count = count

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.count


class FakeTorch:
    """Minimal torch fake for device selection and inference mode."""

    def __init__(
        self,
        *,
        cuda_available: bool = False,
        device_count: int = 0,
    ) -> None:
        self.cuda = FakeCuda(cuda_available, device_count)
        self.inference_mode_calls = 0

    def inference_mode(self) -> _InferenceMode:
        self.inference_mode_calls += 1
        return _InferenceMode()


def _make_classifier(
    *,
    config: ECSConfig | None = None,
    tokenizer: FakeTokenizer | None = None,
    model: object | None = None,
    predict_fn: Callable[[object], object] | None = None,
    torch_module: FakeTorch | None = None,
    label_indices: Mapping[StanceLabel, int] | None = None,
) -> NLIStanceClassifier:
    return NLIStanceClassifier(
        config if config is not None else ECSConfig(),
        _tokenizer=(
            tokenizer
            if tokenizer is not None
            else FakeTokenizer()
        ),
        _model=model,
        _predict_fn=predict_fn,
        _torch=torch_module,
        _label_indices=label_indices,
    )


def _predict_with_logits(
    logits: object,
    *,
    config: ECSConfig | None = None,
    evidence: list[object] | None = None,
    tokenizer: FakeTokenizer | None = None,
) -> list[StancePrediction]:
    classifier = _make_classifier(
        config=config,
        tokenizer=tokenizer,
        predict_fn=lambda encoded: logits,
        label_indices=DEFAULT_LABEL_INDICES,
    )
    return classifier.predict(
        "claim",
        evidence if evidence is not None else ["evidence"],
    )


def test_maps_all_three_labels_in_default_model_order() -> None:
    rows = [
        [2.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
        [0.0, 1.0, 2.0],
    ]
    classifier = _make_classifier(
        predict_fn=lambda encoded: rows,
        label_indices=DEFAULT_LABEL_INDICES,
    )

    predictions = classifier.predict(
        "claim",
        ["first evidence", "second evidence", "third evidence"],
    )

    assert [item.label for item in predictions] == [
        StanceLabel.REFUTES,
        StanceLabel.SUPPORTS,
        StanceLabel.NOT_ENOUGH_INFO,
    ]
    assert [item.evidence_index for item in predictions] == [0, 1, 2]


@pytest.mark.parametrize(
    ("id2label", "logits", "expected"),
    [
        (
            {0: "entailment", 1: "neutral", 2: "contradiction"},
            [[2.0, 0.0, 0.0]],
            StanceLabel.SUPPORTS,
        ),
        (
            {0: "neutral", 1: "contradiction", 2: "entailment"},
            [[0.0, 2.0, 0.0]],
            StanceLabel.REFUTES,
        ),
    ],
)
def test_maps_alternate_valid_label_orders(
    id2label: Mapping[object, object],
    logits: object,
    expected: StanceLabel,
) -> None:
    model = FakeModel(id2label=id2label)
    classifier = _make_classifier(
        model=model,
        predict_fn=lambda encoded: logits,
    )

    predictions = classifier.predict("claim", ["evidence"])

    assert predictions[0].label is expected


def test_label_mapping_accepts_string_indices_and_semantic_aliases() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            id2label={
                "0": "CONTRADICTION",
                "1": "ENTAILMENT",
                "2": "NOT ENOUGH INFO",
            },
        )
    )
    classifier = _make_classifier(
        model=model,
        predict_fn=lambda encoded: [[0.0, 2.0, 0.0]],
    )

    predictions = classifier.predict("claim", ["evidence"])

    assert predictions[0].label is StanceLabel.SUPPORTS


def test_label2id_fallback_is_used_when_id2label_is_empty() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            id2label={},
            label2id={
                "contradiction": "0",
                "entailment": "1",
                "neutral": "2",
            },
        )
    )
    classifier = _make_classifier(
        model=model,
        predict_fn=lambda encoded: [[0.0, 2.0, 0.0]],
    )

    predictions = classifier.predict("claim", ["evidence"])

    assert predictions[0].label is StanceLabel.SUPPORTS


@pytest.mark.parametrize(
    "id2label",
    [
        {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
        {0: "contradiction", 1: "entailment", 2: "support"},
        {0: "contradiction", 1: "entailment"},
        {
            0: "contradiction",
            1: "entailment",
            2: "neutral",
            3: "other",
        },
        {0: "contradiction", 1: "entailment", 3: "neutral"},
    ],
)
def test_unsupported_or_ambiguous_label_mappings_are_rejected(
    id2label: Mapping[object, object],
) -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(id2label=id2label)
    )
    classifier = _make_classifier(
        model=model,
        predict_fn=lambda encoded: [[0.0, 1.0, 2.0]],
    )

    with pytest.raises(UnsupportedLabelMappingError):
        classifier.predict("claim", ["evidence"])


def test_duplicate_label2id_indices_are_rejected() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            id2label={},
            label2id={
                "contradiction": 0,
                "entailment": 0,
                "neutral": 2,
            },
        )
    )
    classifier = _make_classifier(
        model=model,
        predict_fn=lambda encoded: [[0.0, 1.0, 2.0]],
    )

    with pytest.raises(UnsupportedLabelMappingError):
        classifier.predict("claim", ["evidence"])


def test_pair_order_is_evidence_then_claim() -> None:
    tokenizer = FakeTokenizer()
    classifier = _make_classifier(
        tokenizer=tokenizer,
        predict_fn=lambda encoded: [[0.0, 1.0, 2.0]] * 2,
        label_indices=DEFAULT_LABEL_INDICES,
    )

    classifier.predict(
        "the claim",
        ["first evidence", "second evidence"],
    )

    batch_calls = [
        call
        for call in tokenizer.calls
        if isinstance(call[0], list)
    ]
    assert len(batch_calls) == 1
    first, second, kwargs = batch_calls[0]
    assert first == ["first evidence", "second evidence"]
    assert second == ["the claim", "the claim"]
    assert kwargs["padding"] is True
    assert kwargs["truncation"] == "only_first"
    assert kwargs["max_length"] == 512
    assert kwargs["return_tensors"] == "pt"

    claim_calls = [
        call
        for call in tokenizer.calls
        if call[0] == "" and call[1] == "the claim"
    ]
    assert claim_calls
    assert claim_calls[0][2]["truncation"] is False


def test_softmax_probabilities_and_confidence() -> None:
    logits = [0.0, 1.0, -1.0]
    predictions = _predict_with_logits([logits])

    exponentials = [
        math.exp(0.0),
        math.exp(1.0),
        math.exp(-1.0),
    ]
    total = sum(exponentials)
    expected = [value / total for value in exponentials]

    prediction = predictions[0]
    assert prediction.label is StanceLabel.SUPPORTS
    assert prediction.confidence == pytest.approx(max(expected))
    assert prediction.probabilities.refutes == pytest.approx(expected[0])
    assert prediction.probabilities.supports == pytest.approx(expected[1])
    assert prediction.probabilities.not_enough_info == pytest.approx(
        expected[2]
    )
    assert prediction.abstained is False
    assert prediction.truncated is False


def test_confidence_threshold_abstains_and_preserves_outputs() -> None:
    config = ECSConfig(confidence_threshold=0.8)
    predictions = _predict_with_logits(
        [[0.0, 0.0, 0.0]],
        config=config,
    )

    prediction = predictions[0]
    assert prediction.label is StanceLabel.NOT_ENOUGH_INFO
    assert prediction.abstained is True
    assert prediction.confidence == pytest.approx(1.0 / 3.0)
    assert prediction.probabilities.supports == pytest.approx(1.0 / 3.0)
    assert prediction.probabilities.refutes == pytest.approx(1.0 / 3.0)
    assert prediction.probabilities.not_enough_info == pytest.approx(
        1.0 / 3.0
    )


def test_threshold_none_uses_argmax() -> None:
    predictions = _predict_with_logits([[0.0, 0.0, 0.0]])

    prediction = predictions[0]
    assert prediction.label is StanceLabel.SUPPORTS
    assert prediction.abstained is False
    assert prediction.confidence == pytest.approx(1.0 / 3.0)


def test_empty_evidence_returns_without_touching_resources() -> None:
    tokenizer = Mock()
    model = Mock()
    classifier = NLIStanceClassifier(
        ECSConfig(),
        _tokenizer=tokenizer,
        _model=model,
        _torch=FakeTorch(),
    )

    assert classifier.predict("claim", []) == []
    assert classifier.predict("claim", iter(())) == []
    tokenizer.assert_not_called()
    model.assert_not_called()


def test_lazy_loading_reuses_resources_and_passes_loading_options() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel(logits=[[0.0, 1.0, 2.0]])
    tokenizer_loader = Mock(return_value=tokenizer)
    model_loader = Mock(return_value=model)
    fake_transformers = MagicMock()
    fake_transformers.AutoTokenizer.from_pretrained = tokenizer_loader
    fake_transformers.AutoModelForSequenceClassification.from_pretrained = (
        model_loader
    )
    fake_torch = FakeTorch(cuda_available=False)
    config = ECSConfig(
        model_name="local/model",
        model_revision="rev-1",
        local_files_only=True,
    )

    with patch.dict(
        sys.modules,
        {"torch": fake_torch, "transformers": fake_transformers},
    ):
        classifier = NLIStanceClassifier(config)
        assert tokenizer_loader.call_count == 0
        assert model_loader.call_count == 0

        assert classifier.predict("claim", []) == []
        assert tokenizer_loader.call_count == 0
        assert model_loader.call_count == 0

        first = classifier.predict("claim", ["evidence"])
        assert first[0].label is StanceLabel.NOT_ENOUGH_INFO
        assert tokenizer_loader.call_count == 1
        assert model_loader.call_count == 1
        tokenizer_loader.assert_called_once_with(
            "local/model",
            revision="rev-1",
            local_files_only=True,
        )
        model_loader.assert_called_once_with(
            "local/model",
            revision="rev-1",
            local_files_only=True,
        )
        assert "trust_remote_code" not in tokenizer_loader.call_args.kwargs
        assert "trust_remote_code" not in model_loader.call_args.kwargs
        assert model.eval_calls == 1
        assert model.to_calls == ["cpu"]
        assert fake_torch.inference_mode_calls == 1

        second = classifier.predict("claim", ["another evidence"])
        assert len(second) == 1
        assert tokenizer_loader.call_count == 1
        assert model_loader.call_count == 1
        assert model.eval_calls == 1
        assert fake_torch.inference_mode_calls == 2


def test_tokenizer_load_failure_is_wrapped_and_chained() -> None:
    boom = OSError("tokenizer boom")
    fake_transformers = MagicMock()
    fake_transformers.AutoTokenizer.from_pretrained = Mock(
        side_effect=boom
    )

    with patch.dict(
        sys.modules,
        {"transformers": fake_transformers},
    ):
        classifier = NLIStanceClassifier(ECSConfig())
        with pytest.raises(StanceModelError) as exc_info:
            classifier.predict("claim", ["evidence"])

    assert exc_info.value.__cause__ is boom


def test_model_load_failure_is_wrapped_and_chained() -> None:
    boom = OSError("model boom")
    fake_transformers = MagicMock()
    fake_transformers.AutoTokenizer.from_pretrained = Mock(
        return_value=FakeTokenizer()
    )
    fake_transformers.AutoModelForSequenceClassification.from_pretrained = (
        Mock(side_effect=boom)
    )

    with patch.dict(
        sys.modules,
        {
            "torch": FakeTorch(),
            "transformers": fake_transformers,
        },
    ):
        classifier = NLIStanceClassifier(ECSConfig())
        with pytest.raises(StanceModelError) as exc_info:
            classifier.predict("claim", ["evidence"])

    assert exc_info.value.__cause__ is boom


def test_missing_transformers_backend_is_wrapped() -> None:
    with patch.dict(sys.modules, {"transformers": None}):
        classifier = NLIStanceClassifier(ECSConfig())
        with pytest.raises(StanceModelError):
            classifier.predict("claim", ["evidence"])


def test_missing_torch_backend_is_wrapped() -> None:
    classifier = NLIStanceClassifier(
        ECSConfig(),
        _tokenizer=FakeTokenizer(),
        _model=FakeModel(),
    )

    with patch.dict(sys.modules, {"torch": None}):
        with pytest.raises(StanceModelError):
            classifier.predict("claim", ["evidence"])


def test_inference_failure_is_wrapped_and_chained() -> None:
    boom = RuntimeError("inference boom")

    def fail(encoded: object) -> object:
        raise boom

    classifier = _make_classifier(
        predict_fn=fail,
        label_indices=DEFAULT_LABEL_INDICES,
    )

    with pytest.raises(StanceInferenceError) as exc_info:
        classifier.predict("claim", ["evidence"])

    assert exc_info.value.__cause__ is boom


def test_model_call_failure_is_wrapped_and_chained() -> None:
    boom = RuntimeError("model boom")
    model = FakeModel(fail=boom)
    classifier = _make_classifier(
        model=model,
        torch_module=FakeTorch(),
    )

    with pytest.raises(StanceInferenceError) as exc_info:
        classifier.predict("claim", ["evidence"])

    assert exc_info.value.__cause__ is boom


@pytest.mark.parametrize(
    "logits",
    [
        [[float("nan"), 0.0, 0.0]],
        [[float("inf"), 0.0, 0.0]],
        [[0.0, 1.0]],
        [[0.0, 1.0, 2.0, 3.0]],
        [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        [0.0, 1.0, 2.0],
        [[0.0, "1.0", 2.0]],
        None,
    ],
)
def test_invalid_logits_are_rejected(logits: object) -> None:
    classifier = _make_classifier(
        predict_fn=lambda encoded: logits,
        label_indices=DEFAULT_LABEL_INDICES,
    )

    with pytest.raises(StanceInferenceError):
        classifier.predict("claim", ["evidence"])


def test_claim_too_long_is_rejected_without_truncation() -> None:
    tokenizer = FakeTokenizer()
    classifier = _make_classifier(
        config=ECSConfig(max_length=5),
        tokenizer=tokenizer,
        predict_fn=lambda encoded: [[0.0, 1.0, 2.0]],
        label_indices=DEFAULT_LABEL_INDICES,
    )

    with pytest.raises(StanceInputError, match="Claim requires"):
        classifier.predict("one two three", ["evidence"])

    claim_calls = [
        call
        for call in tokenizer.calls
        if call[0] == "" and call[1] == "one two three"
    ]
    assert claim_calls
    assert claim_calls[0][2]["truncation"] is False


def test_evidence_character_truncation_sets_flag() -> None:
    tokenizer = FakeTokenizer()
    classifier = _make_classifier(
        config=ECSConfig(max_evidence_chars=5),
        tokenizer=tokenizer,
        predict_fn=lambda encoded: [[0.0, 1.0, 2.0]],
        label_indices=DEFAULT_LABEL_INDICES,
    )

    predictions = classifier.predict("claim", ["abcdefgh"])

    assert predictions[0].truncated is True
    batch_calls = [
        call
        for call in tokenizer.calls
        if isinstance(call[0], list)
    ]
    assert batch_calls[0][0] == ["abcde"]


def test_short_evidence_is_not_marked_truncated() -> None:
    predictions = _predict_with_logits(
        [[0.0, 1.0, 2.0]],
        evidence=["abc"],
    )

    assert predictions[0].truncated is False


def test_evidence_token_truncation_sets_flag() -> None:
    classifier = _make_classifier(
        config=ECSConfig(max_length=8),
        predict_fn=lambda encoded: [[0.0, 1.0, 2.0]],
        label_indices=DEFAULT_LABEL_INDICES,
    )

    predictions = classifier.predict(
        "claim",
        ["one two three four five"],
    )

    assert predictions[0].truncated is True


def test_plain_text_and_mapping_evidence_are_supported() -> None:
    tokenizer = FakeTokenizer()
    evidence_record = {
        "title": "title",
        "url": "https://example.test",
        "content": "dict evidence",
    }
    classifier = _make_classifier(
        tokenizer=tokenizer,
        predict_fn=lambda encoded: [
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
        ],
        label_indices=DEFAULT_LABEL_INDICES,
    )

    predictions = classifier.predict(
        "claim",
        ["plain evidence", evidence_record],
    )

    assert [item.evidence_index for item in predictions] == [0, 1]
    assert [item.label for item in predictions] == [
        StanceLabel.REFUTES,
        StanceLabel.SUPPORTS,
    ]
    assert evidence_record == {
        "title": "title",
        "url": "https://example.test",
        "content": "dict evidence",
    }
    batch_calls = [
        call
        for call in tokenizer.calls
        if isinstance(call[0], list)
    ]
    assert batch_calls[0][0] == ["plain evidence", "dict evidence"]


@pytest.mark.parametrize(
    "claim",
    ["", "   ", "\n\t", None, 123, b"bytes"],
)
def test_invalid_claims_are_rejected(claim: object) -> None:
    classifier = _make_classifier(
        predict_fn=lambda encoded: [[0.0, 1.0, 2.0]],
        label_indices=DEFAULT_LABEL_INDICES,
    )

    with pytest.raises(StanceInputError):
        classifier.predict(claim, ["evidence"])


@pytest.mark.parametrize(
    "evidence",
    [
        None,
        123,
        "not a sequence",
        b"bytes",
        {"content": "single mapping"},
        [None],
        [{"title": "missing content"}],
        [{"content": 123}],
        [{"content": "   "}],
        [""],
        ["   "],
    ],
)
def test_invalid_evidence_is_rejected(evidence: object) -> None:
    classifier = _make_classifier(
        predict_fn=lambda encoded: [[0.0, 1.0, 2.0]],
        label_indices=DEFAULT_LABEL_INDICES,
    )

    with pytest.raises(StanceInputError):
        classifier.predict("claim", evidence)


@pytest.mark.parametrize(
    ("cuda_available", "expected_device"),
    [
        (True, "cuda"),
        (False, "cpu"),
    ],
)
def test_auto_device_selection(
    cuda_available: bool,
    expected_device: str,
) -> None:
    model = FakeModel(logits=[[0.0, 1.0, 2.0]])
    fake_torch = FakeTorch(
        cuda_available=cuda_available,
        device_count=1 if cuda_available else 0,
    )
    classifier = _make_classifier(
        model=model,
        torch_module=fake_torch,
    )

    predictions = classifier.predict("claim", ["evidence"])

    assert len(predictions) == 1
    assert model.to_calls == [expected_device]
    assert fake_torch.inference_mode_calls == 1


def test_explicit_cuda_unavailable_is_rejected() -> None:
    model = FakeModel()
    classifier = _make_classifier(
        config=ECSConfig(device="cuda"),
        model=model,
        torch_module=FakeTorch(cuda_available=False),
    )

    with pytest.raises(StanceModelError, match="CUDA"):
        classifier.predict("claim", ["evidence"])

    assert model.eval_calls == 0
    assert model.to_calls == []


def test_explicit_cuda_index_out_of_range_is_rejected() -> None:
    classifier = _make_classifier(
        config=ECSConfig(device="cuda:2"),
        model=FakeModel(),
        torch_module=FakeTorch(
            cuda_available=True,
            device_count=1,
        ),
    )

    with pytest.raises(StanceModelError, match="out of range"):
        classifier.predict("claim", ["evidence"])


def test_explicit_cuda_index_is_passed_to_model() -> None:
    model = FakeModel(logits=[[0.0, 1.0, 2.0]])
    classifier = _make_classifier(
        config=ECSConfig(device="cuda:1"),
        model=model,
        torch_module=FakeTorch(
            cuda_available=True,
            device_count=2,
        ),
    )

    classifier.predict("claim", ["evidence"])

    assert model.to_calls == ["cuda:1"]


def test_batch_size_is_respected_and_order_is_preserved() -> None:
    rows = [
        [2.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
        [0.0, 1.0, 2.0],
    ]
    batch_sizes: list[int] = []

    def predict(encoded: object) -> object:
        batch_size = len(encoded["input_ids"])
        batch_sizes.append(batch_size)
        start = sum(batch_sizes[:-1])
        return rows[start : start + batch_size]

    classifier = _make_classifier(
        config=ECSConfig(batch_size=2),
        predict_fn=predict,
        label_indices=DEFAULT_LABEL_INDICES,
    )

    predictions = classifier.predict(
        "claim",
        ["e0", "e1", "e2"],
    )

    assert batch_sizes == [2, 1]
    assert [item.label for item in predictions] == [
        StanceLabel.REFUTES,
        StanceLabel.SUPPORTS,
        StanceLabel.NOT_ENOUGH_INFO,
    ]
    assert [item.evidence_index for item in predictions] == [0, 1, 2]


def test_tokenizer_model_max_length_is_enforced() -> None:
    classifier = _make_classifier(
        config=ECSConfig(max_length=16),
        tokenizer=FakeTokenizer(model_max_length=8),
        predict_fn=lambda encoded: [[0.0, 1.0, 2.0]],
        label_indices=DEFAULT_LABEL_INDICES,
    )

    with pytest.raises(StanceModelError, match="model_max_length"):
        classifier.predict("claim", ["evidence"])


def test_importing_classifier_module_does_not_load_heavy_backends() -> None:
    code = "\n".join(
        [
            "import sys",
            "import src.evidence_conflict",
            "import src.evidence_conflict.nli_classifier",
            "assert 'torch' not in sys.modules, 'torch was imported'",
            "assert 'transformers' not in sys.modules, "
            "'transformers was imported'",
            "assert 'sentence_transformers' not in sys.modules, "
            "'sentence_transformers was imported'",
            "print('ok')",
        ]
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
