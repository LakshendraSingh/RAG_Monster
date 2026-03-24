from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

class DocumentChunker:
    """Handles splitting large documents into smaller semantic chunks."""
    
    @staticmethod
    def chunk_documents_recursive(
        documents: List[Document], 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ) -> List[Document]:
        """
        Splits documents using Recursive Character splitting.
        This is an excellent default for ensuring context is mostly preserved.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(documents)
        return DocumentChunker._clean_metadata(chunks)

    @staticmethod
    def chunk_documents_semantic(
        documents: List[Document],
        embeddings
    ) -> List[Document]:
        """
        Splits documents based on semantic meaning using an embedding model.
        This adapts chunk boundaries to break where meaning shifts.
        """
        semantic_splitter = SemanticChunker(embeddings)
        chunks = semantic_splitter.split_documents(documents)
        return DocumentChunker._clean_metadata(chunks)

    @staticmethod
    def _clean_metadata(chunks: List[Document]) -> List[Document]:
        """Ensures metadata keys satisfy vector DB requirements."""
        for chunk in chunks:
            clean_metadata = {}
            for key, value in chunk.metadata.items():
                # Replace invalid characters in the key
                safe_key = str(key).replace(".", "_").replace("-", "_").replace(" ", "_").replace(":", "_")
                
                # Simple primitive types for metadata
                if isinstance(value, (dict, list)):
                    clean_metadata[safe_key] = str(value)
                else:
                    clean_metadata[safe_key] = value
            chunk.metadata = clean_metadata
        return chunks
