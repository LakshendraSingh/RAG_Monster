from langchain_community.embeddings import OllamaEmbeddings
from config import EMBEDDING_MODEL_NAME, OLLAMA_BASE_URL

class Embedder:
    """Manages the creation and configuration of Embedding Models."""
    
    _instance = None
    
    @staticmethod
    def get_embeddings():
        """Singleton pattern for embedding model to avoid reloading into memory multiple times."""
        if Embedder._instance is None:
            print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} via Ollama...")
            Embedder._instance = OllamaEmbeddings(
                base_url=OLLAMA_BASE_URL,
                model=EMBEDDING_MODEL_NAME
            )
        return Embedder._instance
