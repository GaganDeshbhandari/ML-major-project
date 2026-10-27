"""Cross-encoder reranking for evidence candidates.

Model: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` by default (overridable
via the ``RERANK_MODEL`` environment variable or constructor argument).

Loading strategy: importing this module never imports
``sentence-transformers`` and never downloads weights. The model loads
lazily on the first :meth:`CrossEncoderReranker.score` call that has no
injected ``_predict_fn``/``_model``, and the instance caches it for all
later calls. :func:`get_default_reranker` additionally caches one shared
instance process-wide.

Failures (missing backend, download errors, inference errors,
non-numeric or length-mismatched scores) raise
:class:`src.retrieval.evidence_retriever.RerankerError` chained with
``from``. There is intentionally no silent fallback to Tavily ordering.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Callable, Sequence

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#: Environment variable overriding the model name.
RERANK_MODEL_ENV = "RERANK_MODEL"

#: Maximum candidate characters sent to the cross-encoder per document.
#: Cheap guard before the model's own tokenizer truncation; configurable
#: per instance via ``max_chars``.
MAX_RERANK_CHARS = 2000

__all__ = [
    "DEFAULT_RERANK_MODEL",
    "MAX_RERANK_CHARS",
    "RERANK_MODEL_ENV",
    "CrossEncoderReranker",
    "clear_default_reranker_cache",
    "get_default_reranker",
    "truncate_for_rerank",
]


def truncate_for_rerank(text: str, limit: int = MAX_RERANK_CHARS) -> str:
    """Truncate candidate text to at most ``limit`` characters.

    The evidence text itself is never rewritten — only cut off. The
    model's tokenizer applies its own sub-word truncation afterwards.
    """
    if not isinstance(text, str):
        return ""
    if limit is not None and len(text) > limit:
        return text[:limit]
    return text


class CrossEncoderReranker:
    """Scorer wrapping a sentence-transformers CrossEncoder.

    Args:
        model_name: Checkpoint name. Defaults to ``RERANK_MODEL`` env var,
            then :data:`DEFAULT_RERANK_MODEL`. Resolved lazily at load time.
        max_chars: Per-document character cap applied before scoring.
        _predict_fn: Injected ``predict(pairs)`` for tests. When set, no
            model is ever loaded.
        _model: Injected pre-loaded model for tests.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        max_chars: int = MAX_RERANK_CHARS,
        _predict_fn: Callable[[list[tuple[str, str]]], Sequence[Any]] | None = None,
        _model: Any | None = None,
    ) -> None:
        self._model_name = model_name
        self._max_chars = max_chars
        self._predict_fn = _predict_fn
        self._model = _model

    def _resolve_name(self) -> str:
        return self._model_name or os.getenv(RERANK_MODEL_ENV, DEFAULT_RERANK_MODEL)

    def _get_model(self) -> Any:
        """Load and cache the CrossEncoder on first use."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except Exception as exc:
                from src.retrieval.evidence_retriever import RerankerError

                raise RerankerError(
                    "Reranker backend unavailable "
                    "(sentence-transformers import failed). "
                    "Install it with: pip install 'sentence-transformers>=2.7'."
                ) from exc
            name = self._resolve_name()
            try:
                self._model = CrossEncoder(name)
            except Exception as exc:
                from src.retrieval.evidence_retriever import RerankerError

                raise RerankerError(f"Reranker model load failed ({name}): {exc}") from exc
        return self._model

    def score(self, claim: str, texts: list[str]) -> list[float]:
        """Score each claim–candidate pair; higher means more relevant.

        Returns a list of floats parallel to ``texts``. Empty input
        returns ``[]`` without touching the model.
        """
        from src.retrieval.evidence_retriever import RerankerError

        if not texts:
            return []
        pairs = [(claim, truncate_for_rerank(t, self._max_chars)) for t in texts]
        try:
            if self._predict_fn is not None:
                raw = self._predict_fn(pairs)
            else:
                raw = self._get_model().predict(pairs)
        except RerankerError:
            raise
        except Exception as exc:
            raise RerankerError(f"Reranker inference failed: {exc}") from exc
        try:
            scores = [float(s) for s in raw]
        except (TypeError, ValueError) as exc:
            raise RerankerError(f"Reranker returned non-numeric scores: {exc}") from exc
        if len(scores) != len(texts):
            raise RerankerError(
                f"Reranker returned {len(scores)} scores for {len(texts)} candidates."
            )
        return scores


@functools.lru_cache(maxsize=1)
def get_default_reranker() -> CrossEncoderReranker:
    """Return the shared reranker instance (constructed, not loaded).

    Safe to call at request time; the underlying model still loads lazily
    on first ``score()``. Use :func:`clear_default_reranker_cache` in tests.
    """
    return CrossEncoderReranker()


def clear_default_reranker_cache() -> None:
    """Clear the :func:`get_default_reranker` cache (mainly for tests)."""
    get_default_reranker.cache_clear()
