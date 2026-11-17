from __future__ import annotations

import logging
from typing import List, Optional

try:
    from langchain_core.documents import Document
except ImportError:
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

from core.vdb import VectorDB
from core.reranker import ReRanker
from core.bm25 import BM25Index

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    rankings: list[list[Document]],
    k: int = 60,
    top_k: int = 5,
) -> list[Document]:
    """
    Reciprocal Rank Fusion (RRF) combines multiple ranked lists into one.

    Score formula:
        RRF_Score(d) = sum_{m in models} 1.0 / (k + rank_m(d))
    where k is a smoothing constant (default: 60).
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank_list in rankings:
        for rank, doc in enumerate(rank_list, start=1):
            key = doc.metadata.get("chunk_id") or doc.page_content
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + (1.0 / (k + rank))

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys[:top_k]]


class RAGRetriever:
    """
    Modular retriever supporting Dense (vector), Sparse (BM25), and Hybrid (RRF)
    retrieval strategies, with CrossEncoder re-ranking.
    """

    def __init__(self, use_reranker: bool = True):
        self.vdb = VectorDB()
        self.base_retriever = self.vdb.get_retriever(search_kwargs={"k": 10})
        self.bm25_index = BM25Index()
        self.use_reranker = use_reranker
        if self.use_reranker:
            self.reranker = ReRanker()

    def add_documents(self, documents: List[Document]):
        """Maintain the sparse BM25 index alongside vector store."""
        self.bm25_index.add_documents(documents)

    def clear(self):
        """Clear the in-memory sparse index."""
        self.bm25_index.clear()

    def _retrieve_dense(self, query: str, top_k: int = 10) -> List[Document]:
        """Perform dense semantic search via VectorDB, falling back to sparse if empty."""
        try:
            retriever = self.vdb.get_retriever(search_kwargs={"k": top_k})
            results = retriever.invoke(query)
            if results:
                return results
            # Fallback to BM25 if dense index is empty or unpopulated
            return self._retrieve_sparse(query, top_k=top_k)
        except Exception as exc:
            logger.warning("dense_retrieval_error", extra={"error": str(exc)})
            return self._retrieve_sparse(query, top_k=top_k)

    def _retrieve_sparse(self, query: str, top_k: int = 10) -> List[Document]:
        """Perform sparse lexical search via BM25."""
        return self.bm25_index.search(query, top_k=top_k)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "hybrid",
    ) -> List[Document]:
        """
        Retrieve documents according to the requested strategy:
          - "dense":  vector similarity search only
          - "sparse": BM25 lexical search only
          - "hybrid": dense + sparse combined via Reciprocal Rank Fusion (RRF)

        If re-ranking is enabled, candidates are scored and re-ranked with CrossEncoder.
        """
        fetch_k = max(top_k * 2, 10)

        if strategy == "dense":
            candidates = self._retrieve_dense(query, top_k=fetch_k)
        elif strategy == "sparse":
            candidates = self._retrieve_sparse(query, top_k=fetch_k)
        else:  # hybrid
            dense_docs = self._retrieve_dense(query, top_k=fetch_k)
            sparse_docs = self._retrieve_sparse(query, top_k=fetch_k)

            if dense_docs and sparse_docs:
                candidates = reciprocal_rank_fusion([dense_docs, sparse_docs], k=60, top_k=fetch_k)
            elif dense_docs:
                candidates = dense_docs
            else:
                candidates = sparse_docs

        if self.use_reranker and candidates:
            return self.reranker.rerank(query, candidates, top_k)

        return candidates[:top_k]

    def retrieve_and_format(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "hybrid",
    ) -> str:
        """Retrieve and format as a plain context string."""
        docs = self.retrieve(query, top_k=top_k, strategy=strategy)
        return "\n\n".join([f"Source Content:\n{doc.page_content}" for doc in docs])
