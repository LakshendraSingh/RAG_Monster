from __future__ import annotations

import hashlib
import os
import re
from typing import List, Union

try:
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_experimental.text_splitter import SemanticChunker
except ImportError:
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    RecursiveCharacterTextSplitter = None
    SemanticChunker = None

from core.code_chunker import CodeChunker, detect_language


def _fallback_split(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Pure Python fallback splitter when langchain_text_splitters is missing."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += max(chunk_size - overlap, 1)
    return [c for c in chunks if c]


class DocumentChunker:
    """
    Handles splitting documents and codebases into logical, semantic chunks
    with deterministic, content-hashed IDs.
    
    Supports:
      - Code files: AST-aware and structural logical chunking (classes, functions, methods)
      - Prose & Documentation: Recursive character splitting or semantic embedding splitting
    """

    _code_chunker = CodeChunker()

    @classmethod
    def chunk_documents_code(
        cls,
        documents: Union[Document, List[Document]],
        max_chunk_size: int = 1500,
        chunk_overlap: int = 200,
    ) -> List[Document]:
        """
        Chunks code documents using AST parsing for Python and structural/bracket
        parsing for JS/TS, Go, Rust, Java, C/C++, etc.
        """
        if not isinstance(documents, (list, tuple)):
            documents = [documents]

        chunker = CodeChunker(
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks: List[Document] = []
        for doc in documents:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        return all_chunks

    @classmethod
    def chunk_documents_recursive(
        cls,
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[Document]:
        """
        Splits documents using Recursive Character splitting for prose,
        automatically delegating code files to AST-aware logical code chunking.
        """
        final_chunks: List[Document] = []
        text_docs: List[Document] = []

        for doc in documents:
            source = doc.metadata.get("source") or doc.metadata.get("file_path") or ""
            lang = doc.metadata.get("language") or detect_language(source)
            if lang and lang != "text":
                # Route code file to AST/syntax-aware chunker
                code_chunks = cls.chunk_documents_code(
                    doc, max_chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                final_chunks.extend(code_chunks)
            else:
                text_docs.append(doc)

        if text_docs:
            if RecursiveCharacterTextSplitter is not None:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    is_separator_regex=False,
                )
                chunks = text_splitter.split_documents(text_docs)
            else:
                chunks = []
                for doc in text_docs:
                    texts = _fallback_split(doc.page_content, chunk_size, chunk_overlap)
                    for t in texts:
                        chunks.append(Document(page_content=t, metadata=dict(doc.metadata)))

            final_chunks.extend(cls._clean_metadata(chunks))

        return final_chunks

    @classmethod
    def chunk_documents_semantic(
        cls,
        documents: List[Document],
        embeddings,
    ) -> List[Document]:
        """
        Splits documents based on semantic meaning using an embedding model.
        Code documents are automatically handled with AST-aware chunking.
        """
        final_chunks: List[Document] = []
        text_docs: List[Document] = []

        for doc in documents:
            source = doc.metadata.get("source") or doc.metadata.get("file_path") or ""
            lang = doc.metadata.get("language") or detect_language(source)
            if lang and lang != "text":
                code_chunks = cls.chunk_documents_code(doc)
                final_chunks.extend(code_chunks)
            else:
                text_docs.append(doc)

        if text_docs:
            if SemanticChunker is not None and embeddings is not None:
                try:
                    semantic_splitter = SemanticChunker(embeddings)
                    chunks = semantic_splitter.split_documents(text_docs)
                    final_chunks.extend(cls._clean_metadata(chunks))
                    return final_chunks
                except Exception:
                    pass
            final_chunks.extend(cls.chunk_documents_recursive(text_docs))

        return final_chunks

    @staticmethod
    def _clean_metadata(chunks: List[Document]) -> List[Document]:
        """
        Cleans metadata and assigns deterministic, content-hashed chunk IDs.
        Ensures idempotent upserts when re-indexing documents.
        """
        for idx, chunk in enumerate(chunks):
            clean_metadata = {}
            for key, value in chunk.metadata.items():
                safe_key = str(key).replace(".", "_").replace("-", "_").replace(" ", "_").replace(":", "_")
                if isinstance(value, (dict, list)):
                    clean_metadata[safe_key] = str(value)
                else:
                    clean_metadata[safe_key] = value

            # Assign deterministic chunk_id based on source file + chunk index + content hash
            source = clean_metadata.get("source") or clean_metadata.get("doc_id") or "doc"
            src_name = os.path.basename(str(source))
            content_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[:12]
            clean_metadata["chunk_id"] = f"{src_name}_{idx}_{content_hash}"
            clean_metadata["chunk_index"] = idx

            chunk.metadata = clean_metadata
        return chunks
