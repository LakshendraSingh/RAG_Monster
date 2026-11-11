from __future__ import annotations

import os
from config import EMBEDDING_MODEL_NAME, OLLAMA_BASE_URL

class Embedder:
    """Manages creation and configuration of Embedding Models (Ollama, HuggingFace, OpenAI, Gemini)."""
    
    _instance = None
    
    @staticmethod
    def get_embeddings(model_name: str | None = None):
        """Singleton pattern for embedding model to avoid reloading into memory multiple times."""
        name = model_name or EMBEDDING_MODEL_NAME
        
        if Embedder._instance is None:
            clean_name = name.lower()
            
            # 1. HuggingFace / Sentence-Transformers
            if clean_name.startswith(("huggingface:", "hf:", "sentence-transformers/")) or "minilm" in clean_name or "bge" in clean_name:
                hf_model = name.split(":", 1)[1] if ":" in name else name
                print(f"Loading HuggingFace embedding model: {hf_model}...")
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    Embedder._instance = HuggingFaceEmbeddings(model_name=hf_model)
                except ImportError:
                    from langchain_huggingface import HuggingFaceEmbeddings
                    Embedder._instance = HuggingFaceEmbeddings(model_name=hf_model)

            # 2. OpenAI Embeddings
            elif clean_name.startswith(("openai:", "text-embedding-")):
                openai_model = name.split(":", 1)[1] if ":" in name else name
                print(f"Loading OpenAI embedding model: {openai_model}...")
                from langchain_community.embeddings import OpenAIEmbeddings
                Embedder._instance = OpenAIEmbeddings(model=openai_model)

            # 3. Default: Ollama Embeddings
            else:
                ollama_model = name.split(":", 1)[1] if clean_name.startswith("ollama:") else name
                print(f"Loading embedding model: {ollama_model} via Ollama ({OLLAMA_BASE_URL})...")
                from langchain_community.embeddings import OllamaEmbeddings
                Embedder._instance = OllamaEmbeddings(
                    base_url=OLLAMA_BASE_URL,
                    model=ollama_model
                )
                
        return Embedder._instance
