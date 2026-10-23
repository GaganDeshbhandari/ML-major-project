"""Unit tests for src.factllama.predictor.

Generation is stubbed or uses fake tensors/tokenizers, so no GPU, network,
authentication or model weights are required.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import src.factllama.predictor as pr
from src.factllama.model_loader import FactLLaMAConfig
from src.factllama.prompt_builder import PromptBuildError


class _StubGenerator:
    def __init__(self, output="Verdict: true", error=None):
        self.output = output
        self.error = error
        self.calls = []

    def generate(self, messages):
        self.calls.append(list(messages))
        if self.error is not None:
            raise self.error
        return self.output


def test_predict_returns_parsed_verdict_and_raw_output():
    generator = _StubGenerator("Verdict: half")
    predictor = pr.FactLLaMAPredictor(generator=generator)

    result = predictor.predict("claim text", ["evidence one"])

    assert result.claim == "claim text"
    assert result.verdict == "half"
    assert result.raw_output == "Verdict: half"
    assert len(generator.calls) == 1


def test_predict_builds_system_and_user_messages():
    generator = _StubGenerator()
    predictor = pr.FactLLaMAPredictor(generator=generator)

    predictor.predict("claim text", ["evidence one", "evidence two"])

    messages = generator.calls[0]
    assert messages[0]["role"] == "system"
    assert "outside knowledge" in messages[0]["content"].lower()
    assert messages[1]["role"] == "user"
    assert "1. evidence one" in messages[1]["content"]
    assert "2. evidence two" in messages[1]["content"]
    assert messages[1]["content"].rstrip().endswith("Verdict:")


def test_unparseable_output_yields_none_verdict():
    generator = _StubGenerator("I am not sure about this one.")
    predictor = pr.FactLLaMAPredictor(generator=generator)

    result = predictor.predict("claim", ["evidence"])

    assert result.verdict is None
    assert result.raw_output == "I am not sure about this one."


def test_invalid_input_does_not_call_generator():
    generator = _StubGenerator()
    predictor = pr.FactLLaMAPredictor(generator=generator)

    with pytest.raises(PromptBuildError):
        predictor.predict("   ", ["evidence"])

    assert generator.calls == []


def test_predict_batch_preserves_order():
    generator = _StubGenerator("true")
    predictor = pr.FactLLaMAPredictor(generator=generator)

    results = predictor.predict_batch(
        [("first", ["e1"]), ("second", ["e2"])]
    )

    assert [result.claim for result in results] == ["first", "second"]


def test_loader_is_not_called_until_predict():
    loader = MagicMock()
    loaded = SimpleNamespace(
        model=object(),
        tokenizer=object(),
        config=FactLLaMAConfig(),
        adapter_path=None,
    )
    loader.load.return_value = loaded
    stub = _StubGenerator("true")

    with patch.object(pr, "LlamaTextGenerator", return_value=stub) as factory:
        predictor = pr.FactLLaMAPredictor(loader=loader)
        loader.load.assert_not_called()

        predictor.predict("claim", ["evidence"])

    loader.load.assert_called_once()
    factory.assert_called_once_with(
        loaded.model,
        loaded.tokenizer,
        max_new_tokens=loaded.config.max_new_tokens,
        do_sample=loaded.config.do_sample,
        temperature=loaded.config.temperature,
    )


def test_loader_and_config_together_are_rejected():
    with pytest.raises(ValueError):
        pr.FactLLaMAPredictor(loader=MagicMock(), config=FactLLaMAConfig())


class _FakeTensor:
    def __init__(self, data):
        self.data = list(data)
        self.shape = (1, len(self.data))
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def __getitem__(self, item):
        if isinstance(item, int):
            # Single-row fake tensor: [0] returns the same 1-D row.
            return self
        return _FakeTensor(self.data[item])


class _FakeTokenizer:
    def __init__(self, *, chat_template="template"):
        self.chat_template = chat_template
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.template_calls = []
        self.prompt = None
        self.decoded = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.template_calls.append(
            (messages, tokenize, add_generation_prompt)
        )
        return "PROMPT"

    def __call__(self, prompt, return_tensors):
        self.prompt = prompt
        return {
            "input_ids": _FakeTensor([1, 2, 3]),
            "attention_mask": _FakeTensor([1, 1, 1]),
        }

    def decode(self, token_ids, skip_special_tokens):
        self.decoded = token_ids.data
        return "Verdict: true"


class _FakeModel:
    def __init__(self):
        self.device = "cuda:0"
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return _FakeTensor([1, 2, 3, 9])


class _FakeTorch:
    def inference_mode(self):
        return nullcontext()


def test_generator_uses_chat_template_and_decodes_new_tokens_only():
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    generator = pr.LlamaTextGenerator(
        model,
        tokenizer,
        max_new_tokens=8,
        torch_module=_FakeTorch(),
    )
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]

    output = generator.generate(messages)

    assert output == "Verdict: true"
    assert tokenizer.template_calls == [(messages, False, True)]
    assert tokenizer.prompt == "PROMPT"
    assert tokenizer.decoded == [9]
    assert model.kwargs["max_new_tokens"] == 8
    assert model.kwargs["do_sample"] is False
    assert "temperature" not in model.kwargs
    assert model.kwargs["pad_token_id"] == 0
    assert model.kwargs["input_ids"].device == "cuda:0"
    assert model.kwargs["attention_mask"].device == "cuda:0"


def test_generator_passes_temperature_only_when_sampling():
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    generator = pr.LlamaTextGenerator(
        model,
        tokenizer,
        max_new_tokens=4,
        do_sample=True,
        temperature=0.7,
        torch_module=_FakeTorch(),
    )

    generator.generate([{"role": "user", "content": "u"}])

    assert model.kwargs["do_sample"] is True
    assert model.kwargs["temperature"] == 0.7


def test_generator_falls_back_when_no_chat_template():
    tokenizer = _FakeTokenizer(chat_template=None)
    model = _FakeModel()
    generator = pr.LlamaTextGenerator(
        model, tokenizer, torch_module=_FakeTorch()
    )

    generator.generate(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]
    )

    assert tokenizer.template_calls == []
    assert "### System:" in tokenizer.prompt
    assert "### Assistant:" in tokenizer.prompt


def test_generator_rejects_empty_messages():
    generator = pr.LlamaTextGenerator(
        _FakeModel(), _FakeTokenizer(), torch_module=_FakeTorch()
    )

    with pytest.raises(pr.PredictionError):
        generator.generate([])
