from __future__ import annotations

import concurrent.futures as _cf
import math
from typing import List

try:
    from sentence_transformers import CrossEncoder
    from langchain_core.documents import Document
except ImportError:
    CrossEncoder = None
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}


def sigmoid(x: float) -> float:
    """Normalize raw logit score to 0.0 - 1.0 probability."""
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


class ReRanker:
    """Re-ranks retrieved documents using a Cross-Encoder and writes normalized scores."""

    _instance = None

    def __new__(cls, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if cls._instance is None:
            cls._instance = super(ReRanker, cls).__new__(cls)
            cls._instance.model = None
            if CrossEncoder is not None:
                try:
                    cls._instance.model = CrossEncoder(model_name, device="cpu")
                except Exception:
                    cls._instance.model = None
        return cls._instance

    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """
        Scores and sorts documents based on relevance to the query.
        Writes normalized (0-1) score into doc.metadata['relevance_score'].
        """
        if not documents:
            return []

        if self.model is None:
            # Fallback if CrossEncoder model failed to load or is uninstalled
            for doc in documents:
                doc.metadata.setdefault("relevance_score", 0.75)
            return documents[:top_k]

        pairs = [[query, doc.page_content] for doc in documents]
        with _cf.ThreadPoolExecutor(max_workers=1) as pool:
            scores = pool.submit(self.model.predict, pairs).result()

        # Normalize scores to 0-1 and attach to metadata
        doc_score_pairs = []
        for doc, score in zip(documents, scores):
            norm_score = round(sigmoid(float(score)), 4)
            doc.metadata["relevance_score"] = norm_score
            doc_score_pairs.append((doc, norm_score))

        # Sort documents by score descending
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in doc_score_pairs[:top_k]]
