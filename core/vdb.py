from langchain_chroma import Chroma
import chromadb
from core.embedder import Embedder
from config import CHROMA_PATH
from typing import List
from langchain_core.documents import Document

class VectorDB:
    """Interface to Chroma Vector Database."""
    def __init__(self, collection_name: str = "rag_collection"):
        self.collection_name = collection_name
        self.embeddings = Embedder.get_embeddings()
        
        # Initialize Chroma with persistence
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PATH
        )

    def add_documents(self, documents: List[Document]):
        """Add chunked documents to Chroma."""
        return self.vector_store.add_documents(documents)

    def get_retriever(self, search_kwargs={"k": 5}):
        """Returns the base vector store retriever."""
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def clear_database(self):
        """Delete all documents in the collection."""
        self.vector_store.delete_collection()
        # Re-initialize to ensure it's ready for new data
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PATH
        )

    def delete_collection(self, name: str):
        """Delete a specific collection by name."""
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        client.delete_collection(name)
