"""
core/code_chunker.py
────────────────────
High-performance, AST-aware and syntax-aware logical chunker for large codebases
(supporting repositories with 300,000+ Lines of Code).

Splits code along logical boundaries:
  - Python: Native AST parsing (classes, methods, functions, module headers, imports)
  - JS/TS, Go, Rust, Java, C/C++, etc.: Structural block & declaration parsing
  - Preserves decorators, signatures, docstrings, and line numbers
  - Injects contextual breadcrumbs (# File: ... | Scope: Class > Func | Lines: X-Y)
    so both dense and BM25 retrievers have full contextual awareness.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from langchain_core.documents import Document
except ImportError:
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}


# Supported language mapping by extension
LANGUAGE_EXTENSIONS: Dict[str, str] = {
    # Python
    ".py": "python",
    ".pyi": "python",
    # JavaScript & TypeScript
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    # Systems & Compiled
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    # JVM & Mobile
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".swift": "swift",
    # Backend & Web
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    # Scripting & Query
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".sql": "sql",
    # Markup & Config
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".html": "html",
    ".css": "css",
}

# Comment prefix per language for contextual breadcrumb headers
COMMENT_PREFIXES: Dict[str, str] = {
    "python": "#",
    "bash": "#",
    "ruby": "#",
    "yaml": "#",
    "toml": "#",
    "javascript": "//",
    "typescript": "//",
    "go": "//",
    "rust": "//",
    "c": "//",
    "cpp": "//",
    "java": "//",
    "kotlin": "//",
    "scala": "//",
    "swift": "//",
    "csharp": "//",
    "php": "//",
    "sql": "--",
}


def detect_language(file_path: str) -> Optional[str]:
    """Detect language from file extension or filename."""
    name = os.path.basename(file_path).lower()
    if name == "dockerfile":
        return "dockerfile"
    ext = Path(file_path).suffix.lower()
    return LANGUAGE_EXTENSIONS.get(ext)


class CodeChunker:
    """
    AST- and syntax-aware chunker for source code files.
    Ensures functions, methods, and classes remain coherent units.
    """

    def __init__(
        self,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 150,
        chunk_overlap: int = 200,
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: Document) -> List[Document]:
        """Chunk a single code Document using logical boundaries."""
        file_path = doc.metadata.get("source") or doc.metadata.get("file_path") or "unknown.py"
        language = doc.metadata.get("language") or detect_language(file_path) or "text"

        if language == "python":
            chunks = self._chunk_python(doc.page_content, file_path, doc.metadata)
        elif language in ("javascript", "typescript", "go", "rust", "c", "cpp", "java", "csharp", "php"):
            chunks = self._chunk_structural(doc.page_content, file_path, language, doc.metadata)
        else:
            chunks = self._chunk_line_based(doc.page_content, file_path, language, doc.metadata)

        return self._finalize_chunks(chunks, file_path, language)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Python AST-Aware Logical Chunking
    # ─────────────────────────────────────────────────────────────────────────

    def _chunk_python(self, content: str, file_path: str, base_meta: dict) -> List[Document]:
        """
        Parses Python code using the standard library `ast` module.
        Extracts:
          - Module-level header (imports, docstrings, global constants)
          - Classes (with methods separated if class is large)
          - Top-level functions (sync and async)
        """
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Fallback to structural chunking if AST parsing fails
            return self._chunk_structural(content, file_path, "python", base_meta)

        lines = content.splitlines(keepends=True)
        total_lines = len(lines)
        if not lines:
            return []

        chunks: List[Document] = []
        covered_lines = set()

        # Helper to slice lines (1-indexed to 0-indexed)
        def get_slice(start_l: int, end_l: int) -> str:
            start_idx = max(0, start_l - 1)
            end_idx = min(total_lines, end_l)
            return "".join(lines[start_idx:end_idx]).strip()

        # Helper to get start line including decorators
        def get_node_start(n) -> int:
            if hasattr(n, "decorator_list") and n.decorator_list:
                return min(getattr(n, "lineno", 1), n.decorator_list[0].lineno)
            return getattr(n, "lineno", 1)

        # Step A: Collect top-level definitions
        top_level_nodes = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                top_level_nodes.append(node)

        # Step B: Module Header (imports, docstring, constants before first func/class)
        first_def_line = get_node_start(top_level_nodes[0]) if top_level_nodes else total_lines + 1
        header_text = "".join(lines[:max(0, first_def_line - 1)]).strip()
        if len(header_text) >= self.min_chunk_size:
            meta = dict(base_meta)
            meta.update({
                "chunk_type": "module_header",
                "scope": "module",
                "start_line": 1,
                "end_line": max(1, first_def_line - 1),
            })
            chunks.append(Document(page_content=header_text, metadata=meta))
            for l in range(1, first_def_line):
                covered_lines.add(l)

        # Step C: Iterate through top-level nodes
        for node in top_level_nodes:
            start_line = get_node_start(node)
            end_line = getattr(node, "end_lineno", start_line)

            # 1. Classes
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                class_text = get_slice(start_line, end_line)

                if len(class_text) <= self.max_chunk_size:
                    # Class is compact: emit as a single chunk
                    meta = dict(base_meta)
                    meta.update({
                        "chunk_type": "class",
                        "scope": class_name,
                        "symbol_name": class_name,
                        "start_line": start_line,
                        "end_line": end_line,
                    })
                    chunks.append(Document(page_content=class_text, metadata=meta))
                else:
                    # Class is large: emit class signature/docstring + each method individually
                    method_nodes = [
                        sub for sub in node.body
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ]

                    # Emit Class Header (class def + docstring + class attributes before first method)
                    first_method_line = get_node_start(method_nodes[0]) if method_nodes else end_line + 1
                    class_header_text = get_slice(start_line, first_method_line - 1)
                    if class_header_text:
                        meta = dict(base_meta)
                        meta.update({
                            "chunk_type": "class_header",
                            "scope": class_name,
                            "symbol_name": class_name,
                            "start_line": start_line,
                            "end_line": first_method_line - 1,
                        })
                        chunks.append(Document(page_content=class_header_text, metadata=meta))

                    # Emit individual methods with class breadcrumbs
                    for m in method_nodes:
                        m_start = get_node_start(m)
                        m_end = getattr(m, "end_lineno", m_start)
                        m_text = get_slice(m_start, m_end)
                        scope_name = f"{class_name}.{m.name}"

                        if len(m_text) > self.max_chunk_size:
                            # Split giant method into sub-chunks with signature header
                            sub_chunks = self._split_large_block(
                                m_text, m_start, scope_name, "python", base_meta
                            )
                            chunks.extend(sub_chunks)
                        else:
                            meta = dict(base_meta)
                            meta.update({
                                "chunk_type": "method",
                                "scope": scope_name,
                                "symbol_name": m.name,
                                "class_name": class_name,
                                "start_line": m_start,
                                "end_line": m_end,
                            })
                            chunks.append(Document(page_content=m_text, metadata=meta))

            # 2. Top-level Functions
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_name = node.name
                fn_text = get_slice(start_line, end_line)

                if len(fn_text) > self.max_chunk_size:
                    sub_chunks = self._split_large_block(
                        fn_text, start_line, fn_name, "python", base_meta
                    )
                    chunks.extend(sub_chunks)
                else:
                    meta = dict(base_meta)
                    meta.update({
                        "chunk_type": "function",
                        "scope": fn_name,
                        "symbol_name": fn_name,
                        "start_line": start_line,
                        "end_line": end_line,
                    })
                    chunks.append(Document(page_content=fn_text, metadata=meta))

            for l in range(start_line, end_line + 1):
                covered_lines.add(l)

        # Step D: Capture any remaining uncovered trailing code / scripts
        uncovered = [l for l in range(1, total_lines + 1) if l not in covered_lines]
        if uncovered:
            start_u = uncovered[0]
            end_u = uncovered[-1]
            rem_text = get_slice(start_u, end_u)
            if len(rem_text) >= self.min_chunk_size:
                meta = dict(base_meta)
                meta.update({
                    "chunk_type": "block",
                    "scope": "module_body",
                    "start_line": start_u,
                    "end_line": end_u,
                })
                chunks.append(Document(page_content=rem_text, metadata=meta))

        return chunks if chunks else [Document(page_content=content.strip(), metadata=base_meta)]

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Universal Structural / Bracket-Aware Chunking (JS, TS, Go, Rust, Java)
    # ─────────────────────────────────────────────────────────────────────────

    _STRUCTURAL_PATTERNS = {
        "javascript": re.compile(
            r"^(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+([a-zA-Z0-9_$]+)|"
            r"^(?:export\s+(?:default\s+)?)?class\s+([a-zA-Z0-9_$]+)|"
            r"^(?:export\s+)?(?:const|let|var)\s+([a-zA-Z0-9_$]+)\s*=\s*(?:async\s*)?\(",
            re.MULTILINE,
        ),
        "typescript": re.compile(
            r"^(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+([a-zA-Z0-9_$]+)|"
            r"^(?:export\s+(?:default\s+)?)?class\s+([a-zA-Z0-9_$]+)|"
            r"^(?:export\s+)?interface\s+([a-zA-Z0-9_$]+)|"
            r"^(?:export\s+)?type\s+([a-zA-Z0-9_$]+)\s*=|"
            r"^(?:export\s+)?(?:const|let|var)\s+([a-zA-Z0-9_$]+)\s*=\s*(?:async\s*)?\(",
            re.MULTILINE,
        ),
        "go": re.compile(
            r"^func\s+(?:\([^)]+\)\s+)?([a-zA-Z0-9_]+)\s*\(|"
            r"^type\s+([a-zA-Z0-9_]+)\s+(?:struct|interface)",
            re.MULTILINE,
        ),
        "rust": re.compile(
            r"^(?:pub(?:\([^)]+\))?\s+)?(?:async\s+)?fn\s+([a-zA-Z0-9_]+)|"
            r"^(?:pub(?:\([^)]+\))?\s+)?(?:struct|enum|trait)\s+([a-zA-Z0-9_]+)|"
            r"^impl(?:<[^>]+>)?\s+(?:[a-zA-Z0-9_]+(?:\s+for\s+)?)?([a-zA-Z0-9_]+)",
            re.MULTILINE,
        ),
        "java": re.compile(
            r"^(?:\s*@\w+(?:\([^)]*\))?\s*)*(?:public|protected|private|static|\s)+class\s+([a-zA-Z0-9_]+)|"
            r"^(?:\s*@\w+(?:\([^)]*\))?\s*)*(?:public|protected|private|static|\s)+interface\s+([a-zA-Z0-9_]+)|"
            r"^(?:\s*@\w+(?:\([^)]*\))?\s*)*(?:public|protected|private|static|final|native|synchronized|\s)+[\w<>\[\],\s]+\s+([a-zA-Z0-9_]+)\s*\([^;]*\)\s*\{",
            re.MULTILINE,
        ),
        "c": re.compile(
            r"^(?:struct|union|enum)\s+([a-zA-Z0-9_]+)\s*\{|"
            r"^[\w*]+\s+([a-zA-Z0-9_]+)\s*\([^;]*\)\s*\{",
            re.MULTILINE,
        ),
        "cpp": re.compile(
            r"^(?:class|struct|namespace)\s+([a-zA-Z0-9_]+)|"
            r"^[\w:*&<>]+\s+([a-zA-Z0-9_:]+)\s*\([^;]*\)\s*\{",
            re.MULTILINE,
        ),
    }

    def _chunk_structural(
        self,
        content: str,
        file_path: str,
        language: str,
        base_meta: dict,
    ) -> List[Document]:
        """
        Splits C-style, Go, Rust, and JS/TS code by top-level declarations and balanced braces.
        """
        pattern = self._STRUCTURAL_PATTERNS.get(language)
        if not pattern:
            return self._chunk_line_based(content, file_path, language, base_meta)

        lines = content.splitlines(keepends=True)
        total_lines = len(lines)
        if not lines:
            return []

        chunks: List[Document] = []
        matches = list(pattern.finditer(content))

        if not matches:
            return self._chunk_line_based(content, file_path, language, base_meta)

        # Helper: convert character index to 1-based line number
        def char_to_line(char_pos: int) -> int:
            return content.count("\n", 0, char_pos) + 1

        # File header before first symbol
        first_char = matches[0].start()
        header_text = content[:first_char].strip()
        if len(header_text) >= self.min_chunk_size:
            meta = dict(base_meta)
            meta.update({
                "chunk_type": "module_header",
                "scope": "header",
                "start_line": 1,
                "end_line": max(1, char_to_line(first_char) - 1),
            })
            chunks.append(Document(page_content=header_text, metadata=meta))

        # Track balanced braces for each match
        for idx, match in enumerate(matches):
            start_pos = match.start()
            start_line = char_to_line(start_pos)
            symbol_name = next((g for g in match.groups() if g), "anonymous")

            # Determine end position: either end of balanced braces or next match start
            end_pos = None
            brace_count = 0
            found_open_brace = False

            for c_i in range(start_pos, len(content)):
                ch = content[c_i]
                if ch == "{":
                    brace_count += 1
                    found_open_brace = True
                elif ch == "}":
                    brace_count -= 1
                    if found_open_brace and brace_count <= 0:
                        end_pos = c_i + 1
                        break

            if end_pos is None:
                # If no closing brace (e.g. interfaces or declarations), slice until next match
                if idx + 1 < len(matches):
                    end_pos = matches[idx + 1].start()
                else:
                    end_pos = len(content)

            chunk_text = content[start_pos:end_pos].strip()
            end_line = char_to_line(end_pos)

            if len(chunk_text) > self.max_chunk_size:
                sub_chunks = self._split_large_block(
                    chunk_text, start_line, symbol_name, language, base_meta
                )
                chunks.extend(sub_chunks)
            elif len(chunk_text) >= self.min_chunk_size or not chunks:
                meta = dict(base_meta)
                meta.update({
                    "chunk_type": "declaration",
                    "scope": symbol_name,
                    "symbol_name": symbol_name,
                    "start_line": start_line,
                    "end_line": end_line,
                })
                chunks.append(Document(page_content=chunk_text, metadata=meta))

        return chunks if chunks else [Document(page_content=content.strip(), metadata=base_meta)]

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Line-Based Fallback Splitter
    # ─────────────────────────────────────────────────────────────────────────

    def _chunk_line_based(
        self,
        content: str,
        file_path: str,
        language: str,
        base_meta: dict,
    ) -> List[Document]:
        """Splits code cleanly along logical line boundaries with overlap."""
        lines = content.splitlines(keepends=True)
        chunks: List[Document] = []
        curr_lines: List[str] = []
        curr_chars = 0
        start_line = 1

        for i, line in enumerate(lines, 1):
            curr_lines.append(line)
            curr_chars += len(line)

            if curr_chars >= self.max_chunk_size:
                chunk_str = "".join(curr_lines).strip()
                if chunk_str:
                    meta = dict(base_meta)
                    meta.update({
                        "chunk_type": "line_block",
                        "scope": f"lines_{start_line}_{i}",
                        "start_line": start_line,
                        "end_line": i,
                    })
                    chunks.append(Document(page_content=chunk_str, metadata=meta))

                # Slide window with line overlap
                overlap_chars = 0
                keep_idx = len(curr_lines)
                for r_idx in range(len(curr_lines) - 1, -1, -1):
                    overlap_chars += len(curr_lines[r_idx])
                    if overlap_chars >= self.chunk_overlap:
                        keep_idx = r_idx
                        break

                curr_lines = curr_lines[keep_idx:]
                curr_chars = sum(len(l) for l in curr_lines)
                start_line = i - len(curr_lines) + 1

        if curr_lines:
            chunk_str = "".join(curr_lines).strip()
            if chunk_str:
                meta = dict(base_meta)
                meta.update({
                    "chunk_type": "line_block",
                    "scope": f"lines_{start_line}_{len(lines)}",
                    "start_line": start_line,
                    "end_line": len(lines),
                })
                chunks.append(Document(page_content=chunk_str, metadata=meta))

        return chunks

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers: Splitting Large Blocks & Contextual Breadcrumb Injection
    # ─────────────────────────────────────────────────────────────────────────

    def _split_large_block(
        self,
        block_text: str,
        start_line: int,
        scope_name: str,
        language: str,
        base_meta: dict,
    ) -> List[Document]:
        """Sub-splits a giant method or block into chunks while preserving the scope header."""
        lines = block_text.splitlines(keepends=True)
        sub_chunks = []
        curr_lines = []
        curr_chars = 0
        part_idx = 1
        sub_start = start_line

        for i, line in enumerate(lines):
            curr_lines.append(line)
            curr_chars += len(line)

            if curr_chars >= self.max_chunk_size:
                text = "".join(curr_lines).strip()
                meta = dict(base_meta)
                meta.update({
                    "chunk_type": "sub_function",
                    "scope": f"{scope_name} (part {part_idx})",
                    "symbol_name": scope_name,
                    "start_line": sub_start,
                    "end_line": start_line + i,
                })
                sub_chunks.append(Document(page_content=text, metadata=meta))
                part_idx += 1
                curr_lines = curr_lines[-max(1, len(curr_lines) // 4):]
                curr_chars = sum(len(l) for l in curr_lines)
                sub_start = start_line + i - len(curr_lines) + 1

        if curr_lines:
            text = "".join(curr_lines).strip()
            if text:
                meta = dict(base_meta)
                meta.update({
                    "chunk_type": "sub_function",
                    "scope": f"{scope_name} (part {part_idx})",
                    "symbol_name": scope_name,
                    "start_line": sub_start,
                    "end_line": start_line + len(lines) - 1,
                })
                sub_chunks.append(Document(page_content=text, metadata=meta))

        return sub_chunks

    def _finalize_chunks(
        self,
        chunks: List[Document],
        file_path: str,
        language: str,
    ) -> List[Document]:
        """
        Injects contextual headers and assigns deterministic content-hashed chunk IDs.
        Header Format:
          # File: core/retriever.py | Language: python | Scope: RAGRetriever.retrieve | Lines: 81-125
        """
        comment_prefix = COMMENT_PREFIXES.get(language, "#")
        src_name = os.path.basename(file_path)

        for idx, chunk in enumerate(chunks):
            meta = chunk.metadata
            scope = meta.get("scope", "module")
            s_line = meta.get("start_line", 1)
            e_line = meta.get("end_line", s_line)

            header = (
                f"{comment_prefix} File: {file_path} | Language: {language} | "
                f"Scope: {scope} | Lines: {s_line}-{e_line}\n"
            )

            # Prepend header if not already present
            if not chunk.page_content.startswith(comment_prefix + " File:"):
                chunk.page_content = header + chunk.page_content

            content_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[:12]
            meta["chunk_id"] = f"{src_name}_{idx}_{content_hash}"
            meta["chunk_index"] = idx
            meta["source"] = file_path
            meta["file_path"] = file_path
            meta["language"] = language

        return chunks
