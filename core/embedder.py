from __future__ import annotations

import hashlib
import logging
import os
from config import EMBEDDING_MODEL_NAME, OLLAMA_BASE_URL, GEMINI_API_KEY

logger = logging.getLogger(__name__)


class DirectSentenceTransformerEmbeddings:
    """Wrapper that implements langchain Embeddings interface via sentence-transformers or hash fallback."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            logger.warning("sentence_transformers_direct_unavailable", extra={"error": str(exc)})

    def embed_query(self, text: str) -> list[float]:
        if self.model is not None:
            return [float(x) for x in self.model.encode(text).tolist()]
        # Deterministic 384-dim pseudo-embedding
        return self._hash_vector(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.model is not None:
            return [[float(x) for x in emb.tolist()] for emb in self.model.encode(texts)]
        return [self._hash_vector(t) for t in texts]

    @staticmethod
    def _hash_vector(text: str, dim: int = 384) -> list[float]:
        # Generate reproducible normalized vector from text tokens
        tokens = text.lower().split()
        vec = [0.0] * dim
        for i, t in enumerate(tokens):
            h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
            idx = (h + i) % dim
            vec[idx] += 1.0
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 1e-9:
            vec = [x / norm for x in vec]
        return vec


class Embedder:
    """Manages creation and configuration of Embedding Models (HF, Gemini, OpenAI, Ollama)."""

    _instance = None

    @staticmethod
    def get_embeddings(model_name: str | None = None):
        name = model_name or EMBEDDING_MODEL_NAME

        if Embedder._instance is None:
            clean_name = name.lower()

            # 1. HuggingFace / Sentence-Transformers
            if (
                clean_name.startswith(("huggingface:", "hf:", "sentence-transformers/"))
                or "minilm" in clean_name
                or "bge" in clean_name
            ):
                hf_model = name.split(":", 1)[1] if ":" in name else name
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    Embedder._instance = HuggingFaceEmbeddings(model_name=hf_model)
                except Exception:
                    try:
                        from langchain_huggingface import HuggingFaceEmbeddings
                        Embedder._instance = HuggingFaceEmbeddings(model_name=hf_model)
                    except Exception:
                        Embedder._instance = DirectSentenceTransformerEmbeddings(model_name=hf_model)

            # 2. OpenAI Embeddings
            elif clean_name.startswith(("openai:", "text-embedding-")):
                openai_model = name.split(":", 1)[1] if ":" in name else name
                try:
                    from langchain_community.embeddings import OpenAIEmbeddings
                    Embedder._instance = OpenAIEmbeddings(model=openai_model)
                except Exception:
                    Embedder._instance = DirectSentenceTransformerEmbeddings()

            # 3. Gemini Embeddings
            elif clean_name.startswith(("gemini", "google")):
                try:
                    from langchain_google_genai import GoogleGenerativeAIEmbeddings
                    Embedder._instance = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
                except Exception:
                    Embedder._instance = DirectSentenceTransformerEmbeddings()

            # 4. Ollama Embeddings
            else:
                ollama_model = name.split(":", 1)[1] if clean_name.startswith("ollama:") else name
                try:
                    from langchain_community.embeddings import OllamaEmbeddings
                    Embedder._instance = OllamaEmbeddings(
                        base_url=OLLAMA_BASE_URL,
                        model=ollama_model,
                    )
                except Exception:
                    Embedder._instance = DirectSentenceTransformerEmbeddings()

        return Embedder._instance
