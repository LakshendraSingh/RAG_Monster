"""
core/bm25.py
────────────
BM25 (Best Matching 25) sparse retrieval engine.

Uses `rank_bm25.BM25Okapi` when available, with a full-fidelity pure-Python
fallback implementing the Okapi BM25 ranking function (k1=1.5, b=0.75).
"""

from __future__ import annotations

import math
import re
from typing import Any, List

try:
    from rank_bm25 import BM25Okapi as _RankBM25Okapi
    _HAS_RANK_BM25 = True
except ImportError:
    _HAS_RANK_BM25 = False


def tokenize(text: str) -> list[str]:
    """Simple alphanumeric tokenizer."""
    return re.findall(r"\w+", text.lower())


class PurePythonBM25:
    """Standard Okapi BM25 implementation in pure Python."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lens) / max(self.corpus_size, 1)

        # Compute document frequencies
        self.df: dict[str, int] = {}
        self.doc_freqs: list[dict[str, int]] = []
        for doc in corpus:
            freqs: dict[str, int] = {}
            for token in doc:
                freqs[token] = freqs.get(token, 0) + 1
            self.doc_freqs.append(freqs)
            for token in freqs:
                self.df[token] = self.df.get(token, 0) + 1

        # Compute IDF
        self.idf: dict[str, float] = {}
        for token, freq in self.df.items():
            # Standard Lucene/BM25 IDF formula
            self.idf[token] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for i, freqs in enumerate(self.doc_freqs):
                if token not in freqs:
                    continue
                tf = freqs[token]
                doc_len = self.doc_lens[i]
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(self.avgdl, 1e-6)))
                scores[i] += idf_val * (num / denom)
        return scores


class BM25Index:
    """Manages indexing and querying chunks with BM25."""

    def __init__(self):
        self.documents: list[Any] = []
        self._engine: Any = None

    def index(self, documents: list[Any]):
        """Index a list of Document objects."""
        self.documents = list(documents)
        if not self.documents:
            self._engine = None
            return

        corpus = [tokenize(doc.page_content) for doc in self.documents]
        if _HAS_RANK_BM25:
            self._engine = _RankBM25Okapi(corpus)
        else:
            self._engine = PurePythonBM25(corpus)

    def add_documents(self, documents: list[Any]):
        """Add more documents and rebuild index."""
        # Deduplicate against existing by content or chunk_id
        existing_keys = {
            doc.metadata.get("chunk_id") or doc.page_content
            for doc in self.documents
        }
        for doc in documents:
            key = doc.metadata.get("chunk_id") or doc.page_content
            if key not in existing_keys:
                self.documents.append(doc)
                existing_keys.add(key)
        self.index(self.documents)

    def search(self, query: str, top_k: int = 5) -> list[Any]:
        """Search documents by query string and return top_k Document objects."""
        if not self.documents or not self._engine:
            return []

        tokens = tokenize(query)
        if not tokens:
            return self.documents[:top_k]

        scores = self._engine.get_scores(tokens)
        doc_score_pairs = list(zip(self.documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in doc_score_pairs[:top_k] if score > 0.0] or [
            doc for doc, _ in doc_score_pairs[:top_k]
        ]

    def clear(self):
        self.documents = []
        self._engine = None
