from __future__ import annotations

import logging
from typing import List
from config import CHROMA_PATH
from core.embedder import Embedder

logger = logging.getLogger(__name__)

try:
    from langchain_chroma import Chroma
    import chromadb
    from langchain_core.documents import Document
except ImportError:
    Chroma = None
    chromadb = None
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}


class VectorDB:
    """Interface to Chroma Vector Database with deterministic ID upsert support."""

    def __init__(self, collection_name: str = "rag_collection"):
        self.collection_name = collection_name
        self.embeddings = Embedder.get_embeddings()

        if Chroma is not None and self.embeddings is not None:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=CHROMA_PATH,
            )
        else:
            self.vector_store = None

    def add_documents(self, documents: List[Document]):
        """
        Add chunked documents to Chroma using deterministic IDs for upsert.
        Prevents duplicate chunks upon re-indexing.
        """
        if self.vector_store is None:
            return [doc.metadata.get("chunk_id", str(i)) for i, doc in enumerate(documents)]

        ids = [doc.metadata.get("chunk_id") for doc in documents]
        if all(ids) and len(ids) == len(documents):
            return self.vector_store.add_documents(documents, ids=ids)
        return self.vector_store.add_documents(documents)

    def get_retriever(self, search_kwargs={"k": 5}):
        """Returns the base vector store retriever."""
        if self.vector_store is not None:
            return self.vector_store.as_retriever(search_kwargs=search_kwargs)

        # Fallback dummy retriever
        class _DummyRetriever:
            def invoke(self, query: str):
                return []
        return _DummyRetriever()

    def clear_database(self):
        """Delete all documents in the collection."""
        if self.vector_store is not None:
            self.vector_store.delete_collection()
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=CHROMA_PATH,
            )

    def delete_collection(self, name: str):
        """Delete a specific collection by name."""
        if chromadb is not None:
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            client.delete_collection(name)
