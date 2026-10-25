"""Unit tests for src.retrieval.reranker.

No model downloads, no GPU, no network. The real sentence-transformers
backend is replaced with a fake module or an injected _predict_fn.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.retrieval import reranker as reranker_mod
from src.retrieval.evidence_retriever import RerankerError
from src.retrieval.reranker import (
    CrossEncoderReranker,
    clear_default_reranker_cache,
    get_default_reranker,
    truncate_for_rerank,
)


@pytest.fixture(autouse=True)
def _clean_singleton():
    clear_default_reranker_cache()
    yield
    clear_default_reranker_cache()


def test_truncate_short_text_unchanged():
    assert truncate_for_rerank("hello", limit=2000) == "hello"


def test_truncate_long_text_cut_to_limit():
    text = "x" * 2500
    out = truncate_for_rerank(text, limit=2000)
    assert len(out) == 2000
    assert out == text[:2000]


def test_truncate_custom_limit():
    assert truncate_for_rerank("abcdef", limit=3) == "abc"


def test_score_uses_predict_fn_and_truncates():
    seen = {}

    def fake_predict(pairs):
        seen["pairs"] = pairs
        return [float(i) for i in range(len(pairs))]

    r = CrossEncoderReranker(_predict_fn=fake_predict, max_chars=10)
    scores = r.score("claim", ["y" * 50, "short"])
    assert scores == pytest.approx([0.0, 1.0])
    assert seen["pairs"][0] == ("claim", "y" * 10)
    assert seen["pairs"][1] == ("claim", "short")


def test_score_empty_texts_never_touches_model():
    def boom(pairs):  # pragma: no cover - must not be called
        raise AssertionError("predict must not be called")

    r = CrossEncoderReranker(_predict_fn=boom)
    assert r.score("claim", []) == []


def test_lazy_loading_constructs_once_for_two_calls():
    constructed = {"n": 0}

    class FakeCrossEncoder:
        def __init__(self, name):
            constructed["n"] += 1
            self.name = name

        def predict(self, pairs):
            return [0.5] * len(pairs)

    fake_st = MagicMock()
    fake_st.CrossEncoder = FakeCrossEncoder
    with patch.dict(sys.modules, {"sentence_transformers": fake_st}):
        r = CrossEncoderReranker(model_name="some-model")
        assert constructed["n"] == 0  # lazy: nothing loaded yet
        assert r.score("c", ["a"]) == pytest.approx([0.5])
        assert r.score("c", ["a", "b"]) == pytest.approx([0.5, 0.5])
        assert constructed["n"] == 1  # cached after first load


def test_missing_backend_wrapped_and_chained():
    with patch.dict(sys.modules, {"sentence_transformers": None}):
        r = CrossEncoderReranker(model_name="some-model")
        with pytest.raises(RerankerError) as exc_info:
            r.score("c", ["text"])
    assert isinstance(exc_info.value.__cause__, Exception)


def test_model_init_failure_wrapped():
    fake_st = MagicMock()
    fake_st.CrossEncoder.side_effect = RuntimeError("no weights")
    with patch.dict(sys.modules, {"sentence_transformers": fake_st}):
        r = CrossEncoderReranker(model_name="some-model")
        with pytest.raises(RerankerError):
            r.score("c", ["text"])


def test_inference_failure_wrapped_and_chained():
    boom = RuntimeError("infer boom")

    def bad_predict(pairs):
        raise boom

    r = CrossEncoderReranker(_predict_fn=bad_predict)
    with pytest.raises(RerankerError) as exc_info:
        r.score("c", ["text"])
    assert exc_info.value.__cause__ is boom


def test_non_numeric_scores_raise():
    r = CrossEncoderReranker(_predict_fn=lambda pairs: ["n/a"])
    with pytest.raises(RerankerError):
        r.score("c", ["text"])


def test_length_mismatch_raises():
    r = CrossEncoderReranker(_predict_fn=lambda pairs: [0.1, 0.2])
    with pytest.raises(RerankerError):
        r.score("c", ["only-one"])


def test_default_singleton_cached_and_clearable():
    first = get_default_reranker()
    second = get_default_reranker()
    assert first is second
    clear_default_reranker_cache()
    assert get_default_reranker() is not first
