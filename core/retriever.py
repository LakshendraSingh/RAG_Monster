from core.vdb import VectorDB
from core.reranker import ReRanker

class RAGRetriever:
    """Combines vector retrieval with optional re-ranking."""
    def __init__(self, use_reranker=True):
        self.vdb = VectorDB()
        self.base_retriever = self.vdb.get_retriever(search_kwargs={"k": 10})
        self.use_reranker = use_reranker
        if self.use_reranker:
            self.reranker = ReRanker()

    def retrieve_and_format(self, query: str, top_k: int = 5) -> str:
        """Retrieve documents, optionally re-rank, and format as context string."""
        docs = self.base_retriever.invoke(query)
        
        if self.use_reranker and docs:
            docs = self.reranker.rerank(query, docs, top_k)
        else:
            docs = docs[:top_k]
            
        context = "\n\n".join([f"Source Content:\n{doc.page_content}" for doc in docs])
        return context
