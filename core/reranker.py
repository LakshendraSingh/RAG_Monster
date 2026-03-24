from sentence_transformers import CrossEncoder
from typing import List
from langchain_core.documents import Document

class ReRanker:
    """Re-ranks retrieved documents using a Cross-Encoder."""
    
    _instance = None
    
    def __new__(cls, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if cls._instance is None:
            cls._instance = super(ReRanker, cls).__new__(cls)
            print(f"Loading re-ranker model: {model_name}...")
            cls._instance.model = CrossEncoder(model_name, device="cpu")
        return cls._instance

    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Scores and sorts documents based on relevance to the query."""
        if not documents:
            return []
            
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)
        
        # Sort documents by score descending
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in doc_score_pairs[:top_k]]
