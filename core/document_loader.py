import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from typing import List
from langchain_core.documents import Document

class DocumentLoader:
    """Handles loading documents from various sources."""
    
    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        """Load text from a PDF file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        loader = PyPDFLoader(file_path)
        return loader.load()

    @staticmethod
    def load_text(file_path: str) -> List[Document]:
        """Load text from a standard TXT file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"TXT file not found: {file_path}")
        loader = TextLoader(file_path)
        return loader.load()

    @staticmethod
    def load_url(url: str) -> List[Document]:
        """Load text from a website URL."""
        loader = WebBaseLoader(url)
        return loader.load()

    @staticmethod
    def load_any(path_or_url: str) -> List[Document]:
        """Automatically determines how to load the provided path or URL."""
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            return DocumentLoader.load_url(path_or_url)
        elif path_or_url.endswith(".pdf"):
            return DocumentLoader.load_pdf(path_or_url)
        elif path_or_url.endswith(".txt") or path_or_url.endswith(".md"):
            return DocumentLoader.load_text(path_or_url)
        else:
            raise ValueError(f"Unsupported document format or source: {path_or_url}")
