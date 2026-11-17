"""
core/document_loader.py
───────────────────────
Document loaders for PDF, TXT, MD, DOCX, HTML, and entire directories.
Includes SSRF protection on URLs and directory-traversal / path allowlisting.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
import urllib.parse
from pathlib import Path
from typing import List

try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
    from langchain_core.documents import Document
except ImportError:
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    PyPDFLoader = None
    TextLoader = None
    WebBaseLoader = None

# ─────────────────────────────────────────────────────────────────────────────
# Security helpers: SSRF protection & Path allowlisting
# ─────────────────────────────────────────────────────────────────────────────

def validate_url_safe(url: str):
    """
    Validate that a URL does not point to internal/private IP ranges (SSRF protection).
    Raises ValueError if the destination is private or blocked.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Security error: Only http and https schemes are permitted: {url}")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"Security error: Invalid URL hostname: {url}")

    if hostname in ("localhost", "127.0.0.1", "::1", "metadata.google.internal"):
        raise ValueError(f"Security error: Access to private/internal host '{hostname}' is blocked.")

    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        # If DNS fails or host does not resolve
        return

    for family, _, _, _, sockaddr in addr_info:
        ip_str = sockaddr[0]
        ip = ipaddress.ip_address(ip_str)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError(
                f"Security error: Access to private/internal network IP '{ip_str}' is blocked."
            )


def validate_path_allowed(file_path: str):
    """
    Ensure the path is within allowed directory roots (path traversal protection).
    Allowed roots default to current working directory or ALLOWED_INDEX_PATHS env var.
    """
    real_path = os.path.realpath(file_path)
    allowed_roots_raw = os.getenv("ALLOWED_INDEX_PATHS", os.getcwd())
    allowed_roots = [
        os.path.realpath(p.strip()) for p in allowed_roots_raw.split(",") if p.strip()
    ]

    is_allowed = any(
        real_path == root or real_path.startswith(root + os.sep)
        for root in allowed_roots
    )

    if not is_allowed:
        raise ValueError(
            f"Security error: Path '{file_path}' resolves to '{real_path}', which is outside "
            f"allowed index roots ({allowed_roots})."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Document Loaders
# ─────────────────────────────────────────────────────────────────────────────

class DocumentLoader:
    """Handles loading documents from files, directories, and URLs."""

    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        validate_path_allowed(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if PyPDFLoader is not None:
            loader = PyPDFLoader(file_path)
            return loader.load()

        # Fallback using pypdf if installed directly
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return [Document(page_content=text, metadata={"source": file_path})]
        except Exception:
            pass

        # Native macOS PDFKit fallback
        import subprocess
        abs_path = os.path.abspath(file_path)
        jxa_script = f'''
ObjC.import("PDFKit");
var url = $.NSURL.fileURLWithPath("{abs_path}");
var doc = $.PDFDocument.alloc.initWithURL(url);
if (doc && doc.string) {{
    doc.string.js;
}} else {{
    "";
}}
'''
        try:
            res = subprocess.run(
                ["osascript", "-l", "JavaScript", "-e", jxa_script],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if res.returncode == 0 and res.stdout.strip():
                return [Document(page_content=res.stdout.strip(), metadata={"source": file_path})]
        except Exception:
            pass

        raise RuntimeError(f"Could not load PDF: {file_path}")

    @staticmethod
    def load_text(file_path: str) -> List[Document]:
        validate_path_allowed(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        return [Document(page_content=text, metadata={"source": file_path})]

    @staticmethod
    def load_docx(file_path: str) -> List[Document]:
        """Load text from a DOCX file."""
        validate_path_allowed(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"DOCX file not found: {file_path}")

        # 1. Try python-docx
        try:
            import docx
            doc = docx.Document(file_path)
            text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
            return [Document(page_content=text, metadata={"source": file_path})]
        except ImportError:
            pass

        # 2. Pure Python zipfile fallback for docx
        import zipfile
        import xml.etree.ElementTree as ET
        try:
            with zipfile.ZipFile(file_path) as z:
                xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            paragraphs = []
            for node in tree.iter():
                if node.tag.endswith("}p"):
                    texts = [child.text for child in node.iter() if child.tag.endswith("}t") and child.text]
                    if texts:
                        paragraphs.append("".join(texts))
            text = "\n".join(paragraphs)
            return [Document(page_content=text, metadata={"source": file_path})]
        except Exception as exc:
            raise RuntimeError(f"Failed to parse docx file: {exc}")

    @staticmethod
    def load_html(file_path_or_url: str) -> List[Document]:
        """Load text from an HTML file or string."""
        if file_path_or_url.startswith(("http://", "https://")):
            return DocumentLoader.load_url(file_path_or_url)

        validate_path_allowed(file_path_or_url)
        with open(file_path_or_url, "r", encoding="utf-8", errors="replace") as f:
            html = f.read()

        # 1. Try BeautifulSoup
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            return [Document(page_content=text, metadata={"source": file_path_or_url})]
        except ImportError:
            pass

        # 2. Pure regex fallback
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return [Document(page_content=text, metadata={"source": file_path_or_url})]

    @staticmethod
    def load_code(file_path: str) -> List[Document]:
        """Load source code from a file and tag with language and line counts."""
        from core.code_chunker import detect_language
        validate_path_allowed(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Code file not found: {file_path}")

        language = detect_language(file_path) or "text"
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()

        line_count = code.count("\n") + (1 if code else 0)
        return [
            Document(
                page_content=code,
                metadata={
                    "source": file_path,
                    "file_path": file_path,
                    "language": language,
                    "line_count": line_count,
                },
            )
        ]

    @staticmethod
    def load_codebase(
        root_path: str,
        max_file_size_kb: int = 1000,
        ignore_patterns: List[str] = None,
        max_workers: int = 8,
    ) -> List[Document]:
        """
        High-performance codebase ingestion designed for large repositories (300k+ LoC).
        Automatically respects .gitignore, ignores binaries, build artifacts, and
        reads files concurrently using a thread pool.
        """
        import fnmatch
        from concurrent.futures import ThreadPoolExecutor
        from core.code_chunker import LANGUAGE_EXTENSIONS, detect_language

        validate_path_allowed(root_path)
        if not os.path.isdir(root_path):
            raise NotADirectoryError(f"Codebase directory not found: {root_path}")

        default_ignores = {
            ".git", ".github", ".idea", ".vscode", "__pycache__", "node_modules",
            "venv", ".venv", "env", ".env", "dist", "build", "target", "vendor",
            "bin", "obj", "pkg", "out", "coverage", ".tox", ".pytest_cache",
            "*.pyc", "*.pyo", "*.so", "*.dylib", "*.dll", "*.exe", "*.class",
            "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Cargo.lock",
            "*.min.js", "*.min.css", "*.map", "*.wasm",
        }
        if ignore_patterns:
            default_ignores.update(ignore_patterns)

        # Parse .gitignore in root_path if available
        gitignore_path = os.path.join(root_path, ".gitignore")
        if os.path.exists(gitignore_path):
            try:
                with open(gitignore_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            default_ignores.add(line.rstrip("/"))
            except Exception:
                pass

        def should_ignore(rel_path: str, filename: str) -> bool:
            parts = rel_path.split(os.sep)
            for part in parts:
                if part in default_ignores or any(fnmatch.fnmatch(part, pat) for pat in default_ignores):
                    return True
            for pat in default_ignores:
                if fnmatch.fnmatch(filename, pat) or fnmatch.fnmatch(rel_path, pat):
                    return True
            return False

        files_to_load = []
        max_bytes = max_file_size_kb * 1024

        for dirpath, dirnames, filenames in os.walk(root_path):
            # Prune ignored directories in-place for speed
            rel_dir = os.path.relpath(dirpath, root_path)
            dirnames[:] = [
                d for d in dirnames
                if not should_ignore(os.path.join(rel_dir, d) if rel_dir != "." else d, d)
            ]

            for fname in filenames:
                rel_fpath = os.path.join(rel_dir, fname) if rel_dir != "." else fname
                if should_ignore(rel_fpath, fname):
                    continue

                full_path = os.path.join(dirpath, fname)
                ext = Path(fname).suffix.lower()

                # Check if it's a code, text, or documentation file
                if ext in LANGUAGE_EXTENSIONS or fname.lower() == "dockerfile" or ext in (".md", ".txt"):
                    try:
                        if os.path.getsize(full_path) <= max_bytes:
                            files_to_load.append(full_path)
                    except OSError:
                        pass

        def _read_file(fpath: str) -> Optional[Document]:
            try:
                lang = detect_language(fpath) or "text"
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    code = f.read()
                line_count = code.count("\n") + (1 if code else 0)
                rel_path = os.path.relpath(fpath, root_path)
                return Document(
                    page_content=code,
                    metadata={
                        "source": fpath,
                        "file_path": fpath,
                        "relative_path": rel_path,
                        "language": lang,
                        "line_count": line_count,
                    },
                )
            except Exception:
                return None

        documents = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for doc in executor.map(_read_file, files_to_load):
                if doc:
                    documents.append(doc)

        return documents

    @staticmethod
    def load_directory(folder_path: str, recursive: bool = True) -> List[Document]:
        """Load all supported documents and code files from a directory."""
        from core.code_chunker import LANGUAGE_EXTENSIONS
        validate_path_allowed(folder_path)
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"Directory not found: {folder_path}")

        path_obj = Path(folder_path)
        pattern = "**/*" if recursive else "*"
        all_docs = []

        for p in path_obj.glob(pattern):
            if not p.is_file():
                continue
            suffix = p.suffix.lower()
            str_path = str(p)
            try:
                if suffix in (".txt", ".md"):
                    all_docs.extend(DocumentLoader.load_text(str_path))
                elif suffix == ".pdf":
                    all_docs.extend(DocumentLoader.load_pdf(str_path))
                elif suffix == ".docx":
                    all_docs.extend(DocumentLoader.load_docx(str_path))
                elif suffix in (".html", ".htm"):
                    all_docs.extend(DocumentLoader.load_html(str_path))
                elif suffix in LANGUAGE_EXTENSIONS or p.name.lower() == "dockerfile":
                    all_docs.extend(DocumentLoader.load_code(str_path))
            except Exception:
                continue

        return all_docs

    @staticmethod
    def load_url(url: str) -> List[Document]:
        """Load text from an external website URL with SSRF check."""
        validate_url_safe(url)
        if WebBaseLoader is not None:
            loader = WebBaseLoader(url)
            return loader.load()

        import urllib.request
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "RAG_MONSTER_Crawler/2.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
        except ImportError:
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()

        return [Document(page_content=text, metadata={"source": url})]

    @staticmethod
    def load_any(path_or_url: str) -> List[Document]:
        """Automatically determines how to load the provided path or URL."""
        from core.code_chunker import LANGUAGE_EXTENSIONS
        if path_or_url.startswith(("http://", "https://")):
            return DocumentLoader.load_url(path_or_url)

        if os.path.isdir(path_or_url):
            return DocumentLoader.load_codebase(path_or_url)

        suffix = Path(path_or_url).suffix.lower()
        if suffix == ".pdf":
            return DocumentLoader.load_pdf(path_or_url)
        elif suffix in (".txt", ".md"):
            return DocumentLoader.load_text(path_or_url)
        elif suffix == ".docx":
            return DocumentLoader.load_docx(path_or_url)
        elif suffix in (".html", ".htm"):
            return DocumentLoader.load_html(path_or_url)
        elif suffix in LANGUAGE_EXTENSIONS or Path(path_or_url).name.lower() == "dockerfile":
            return DocumentLoader.load_code(path_or_url)
        else:
            raise ValueError(f"Unsupported document format or source: {path_or_url}")
