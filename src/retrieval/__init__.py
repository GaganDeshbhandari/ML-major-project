"""Evidence retrieval package (Member 2).

Stage 2: Tavily candidate search (15) + cross-encoder reranking to the
requested top-k (cap 5). ``use_reranker=False`` preserves the Stage 1
retrieval-only baseline. No stance classification, embeddings-based
retrieval, FAISS, or model training here.
"""

from src.retrieval.evidence_retriever import (
    CANDIDATE_LIMIT,
    MAX_RESULTS,
    EvidenceRetrievalError,
    EvidenceSearchError,
    MissingAPIKeyError,
    RerankerError,
    retrieve_evidence,
)
from src.retrieval.reranker import (
    DEFAULT_RERANK_MODEL,
    MAX_RERANK_CHARS,
    CrossEncoderReranker,
    clear_default_reranker_cache,
    get_default_reranker,
    truncate_for_rerank,
)

__all__ = [
    "CANDIDATE_LIMIT",
    "DEFAULT_RERANK_MODEL",
    "MAX_RESULTS",
    "MAX_RERANK_CHARS",
    "CrossEncoderReranker",
    "EvidenceRetrievalError",
    "EvidenceSearchError",
    "MissingAPIKeyError",
    "RerankerError",
    "clear_default_reranker_cache",
    "get_default_reranker",
    "retrieve_evidence",
    "truncate_for_rerank",
]
