"""Unit tests for src.retrieval.evidence_retriever.

All Tavily interactions are mocked: no internet access, no API quota.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.evidence_retriever import (
    EvidenceSearchError,
    MissingAPIKeyError,
    retrieve_evidence,
)

MODULE = "src.retrieval.evidence_retriever"


def _mock_client(response=None, search_error=None, init_error=None):
    """Build a mocked TavilyClient class returning a configured instance."""
    instance = MagicMock()
    if search_error is not None:
        instance.search.side_effect = search_error
    else:
        instance.search.return_value = response
    cls = MagicMock()
    if init_error is not None:
        cls.side_effect = init_error
    else:
        cls.return_value = instance
    return cls, instance


def _with_key():
    """Patch env + load_dotenv so tests never touch the real .env."""
    return (
        patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}),
        patch(f"{MODULE}.load_dotenv", return_value=True),
    )


def test_valid_search_results():
    env_patch, dotenv_patch = _with_key()
    payload = {
        "results": [
            {"title": "Source A", "url": "https://a.example/1", "content": "Evidence A", "score": 0.9},
            {"title": "Source B", "url": "https://b.example/2", "content": "Evidence B", "score": 0.5},
        ]
    }
    cls, instance = _mock_client(response=payload)
    with env_patch, dotenv_patch, patch(f"{MODULE}.TavilyClient", cls):
        out = retrieve_evidence("The Eiffel Tower is in Paris.", top_k=5)
    assert len(out) == 2
    for record in out:
        assert set(record.keys()) == {"title", "url", "content", "relevance_score"}
    assert out[0]["url"] == "https://a.example/1"
    assert out[0]["content"] == "Evidence A"
    assert out[0]["relevance_score"] == pytest.approx(0.9)
    instance.search.assert_called_once()
    _, kwargs = instance.search.call_args
    assert kwargs["query"] == "The Eiffel Tower is in Paris."
    assert kwargs["max_results"] == 5


@pytest.mark.parametrize("bad_claim", ["", "   ", "\n\t  ", None, 123, ["x"]])
@pytest.mark.parametrize("top_k", [5])
def test_empty_and_nonstring_claims(bad_claim, top_k):
    env_patch, dotenv_patch = _with_key()
    cls, _ = _mock_client(response={"results": []})
    with env_patch, dotenv_patch, patch(f"{MODULE}.TavilyClient", cls):
        with pytest.raises(ValueError):
            retrieve_evidence(bad_claim, top_k=top_k)
    cls.assert_not_called()


@pytest.mark.parametrize("bad_top_k", [0, -1, -100, 1.5, "5", None, True, False])
def test_invalid_top_k(bad_top_k):
    env_patch, dotenv_patch = _with_key()
    cls, _ = _mock_client(response={"results": []})
    with env_patch, dotenv_patch, patch(f"{MODULE}.TavilyClient", cls):
        with pytest.raises(ValueError):
            retrieve_evidence("A valid claim.", top_k=bad_top_k)
    cls.assert_not_called()


def test_empty_results():
    env_patch, dotenv_patch = _with_key()
    cls, _ = _mock_client(response={"results": []})
    with env_patch, dotenv_patch, patch(f"{MODULE}.TavilyClient", cls):
        assert retrieve_evidence("Some claim.", top_k=5) == []


def test_duplicate_urls_keep_highest_score():
    env_patch, dotenv_patch = _with_key()
    payload = {
        "results": [
            {"title": "Low", "url": "https://dup.example/x", "content": "Low score", "score": 0.2},
            {"title": "High", "url": "https://dup.example/x", "content": "High score", "score": 0.95},
            {"title": "Other", "url": "https://other.example/y", "content": "Other", "score": 0.5},
        ]
    }
    cls, _ = _mock_client(response=payload)
    with env_patch, dotenv_patch, patch(f"{MODULE}.TavilyClient", cls):
        out = retrieve_evidence("Claim.", top_k=5)
    assert len(out) == 2
    by_url = {r["url"]: r for r in out}
    assert by_url["https://dup.example/x"]["relevance_score"] == pytest.approx(0.95)
    assert by_url["https://dup.example/x"]["content"] == "High score"


def test_missing_urls_and_empty_content_skipped():
    env_patch, dotenv_patch = _with_key()
    payload = {
        "results": [
            {"title": "No URL", "content": "Has content", "score": 0.9},
            {"title": "Empty URL", "url": "   ", "content": "Has content", "score": 0.8},
            {"title": "No content", "url": "https://ok.example/1", "score": 0.7},
            {"title": "Blank content", "url": "https://ok.example/2", "content": "   ", "score": 0.6},
            "not-a-dict",
            {"title": "Good", "url": "https://ok.example/3", "content": "Kept", "score": 0.4},
        ]
    }
    cls, _ = _mock_client(response=payload)
    with env_patch, dotenv_patch, patch(f"{MODULE}.TavilyClient", cls):
        out = retrieve_evidence("Claim.", top_k=5)
    assert len(out) == 1
    assert out[0]["url"] == "https://ok.example/3"
    assert out[0]["content"] == "Kept"


def test_missing_optional_fields_use_defaults():
    env_patch, dotenv_patch = _with_key()
    payload = {
        "results": [
            {"url": "https://notitle.example/1", "content": "No title or score"},
            {"title": None, "url": "https://badscore.example/2", "content": "Bad score", "score": "n/a"},
        ]
    }
    cls, _ = _mock_client(response=payload)
    with env_patch, dotenv_patch, patch(f"{MODULE}.TavilyClient", cls):
        out = retrieve_evidence("Claim.", top_k=5)
    assert len(out) == 2
    assert out[0]["title"] == ""
    assert out[0]["relevance_score"] == pytest.approx(0.0)


def test_sorting_and_top_k_limiting():
    env_patch, dotenv_patch = _with_key()
    payload = {
        "results": [
            {"title": f"S{i}", "url": f"https://s.example/{i}", "content": f"E{i}", "score": s}
            for i, s in enumerate([0.1, 0.9, 0.5, 0.7, 0.3])
        ]
    }
    cls, _ = _mock_client(response=payload)
    with env_patch, dotenv_patch, patch(f"{MODULE}.TavilyClient", cls):
        out = retrieve_evidence("Claim.", top_k=3)
    assert [r["relevance_score"] for r in out] == pytest.approx([0.9, 0.7, 0.5])
    assert len(out) == 3


def test_top_k_capped_at_five():
    env_patch, dotenv_patch = _with_key()
    payload = {
        "results": [
            {"title": f"S{i}", "url": f"https://c.example/{i}", "content": f"E{i}", "score": 1.0 - i * 0.1}
            for i in range(7)
        ]
    }
    cls, instance = _mock_client(response=payload)
    with env_patch, dotenv_patch, patch(f"{MODULE}.TavilyClient", cls):
        out = retrieve_evidence("Claim.", top_k=10)
    assert len(out) == 5
    _, kwargs = instance.search.call_args
    assert kwargs["max_results"] == 5


def test_missing_api_key():
    with patch.dict(os.environ, {}, clear=True), patch(
        f"{MODULE}.load_dotenv", return_value=True
    ), patch(f"{MODULE}.TavilyClient") as cls:
        with pytest.raises(MissingAPIKeyError):
            retrieve_evidence("A claim.", top_k=5)
    cls.assert_not_called()


def test_blank_api_key():
    with patch.dict(os.environ, {"TAVILY_API_KEY": "   "}), patch(
        f"{MODULE}.load_dotenv", return_value=True
    ), patch(f"{MODULE}.TavilyClient") as cls:
        with pytest.raises(MissingAPIKeyError):
            retrieve_evidence("A claim.", top_k=5)
    cls.assert_not_called()


def test_tavily_search_failure():
    env_patch, dotenv_patch = _with_key()
    cls, _ = _mock_client(search_error=RuntimeError("boom"))
    with env_patch, dotenv_patch, patch(f"{MODULE}.TavilyClient", cls):
        with pytest.raises(EvidenceSearchError):
            retrieve_evidence("A claim.", top_k=5)


def test_tavily_client_init_failure():
    env_patch, dotenv_patch = _with_key()
    cls, _ = _mock_client(init_error=ConnectionError("no client"))
    with env_patch, dotenv_patch, patch(f"{MODULE}.TavilyClient", cls):
        with pytest.raises(EvidenceSearchError):
            retrieve_evidence("A claim.", top_k=5)


@pytest.mark.parametrize(
    "bad_response",
    ["oops", 123, {"unexpected": 1, "results": "not-a-list"}],
)
def test_malformed_responses_raise(bad_response):
    env_patch, dotenv_patch = _with_key()
    cls, _ = _mock_client(response=bad_response)
    with env_patch, dotenv_patch, patch(f"{MODULE}.TavilyClient", cls):
        # {"unexpected": 1, "results": "not-a-list"} -> results not a list.
        # {"unexpected": 1} without "results" -> .get default [] -> returns [].
        # Only assert raise for genuinely malformed shapes here.
        if isinstance(bad_response, dict) and "results" not in bad_response:
            assert retrieve_evidence("A claim.", top_k=5) == []
        else:
            with pytest.raises(EvidenceSearchError):
                retrieve_evidence("A claim.", top_k=5)
