import os
import shutil
import tempfile
import unittest
from core.code_chunker import CodeChunker, detect_language
from core.chunker import DocumentChunker
from core.document_loader import DocumentLoader

try:
    from langchain_core.documents import Document
except ImportError:
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}


class TestCodeChunker(unittest.TestCase):

    def setUp(self):
        self.chunker = CodeChunker(max_chunk_size=500, min_chunk_size=50)

    def test_detect_language(self):
        self.assertEqual(detect_language("src/auth/jwt.py"), "python")
        self.assertEqual(detect_language("frontend/App.tsx"), "typescript")
        self.assertEqual(detect_language("server/main.go"), "go")
        self.assertEqual(detect_language("src/lib.rs"), "rust")
        self.assertEqual(detect_language("backend/User.java"), "java")
        self.assertEqual(detect_language("Dockerfile"), "dockerfile")
        self.assertIsNone(detect_language("notes.unknown"))

    def test_python_ast_chunking_classes_and_methods(self):
        sample_py = '''"""Module docstring for auth service."""
import os
import sys

GLOBAL_SECRET = "super-secret"

class TokenManager:
    """Manages token generation and validation."""
    def __init__(self, secret: str):
        self.secret = secret

    def verify_token(self, token: str) -> bool:
        # Check token validity
        if not token:
            return False
        return token.startswith("valid_")

@decorator_one
def generate_token(user_id: str) -> str:
    """Generate a signed user token."""
    return f"token_for_{user_id}"
'''
        doc = Document(page_content=sample_py, metadata={"source": "auth/tokens.py"})
        chunks = self.chunker.chunk_document(doc)

        self.assertGreater(len(chunks), 1)

        # Check module header chunk
        header_chunk = next((c for c in chunks if c.metadata.get("chunk_type") == "module_header"), None)
        self.assertIsNotNone(header_chunk)
        self.assertIn("import os", header_chunk.page_content)
        self.assertIn("GLOBAL_SECRET", header_chunk.page_content)

        # Check class or method chunks
        method_or_class = next(
            (c for c in chunks if "TokenManager" in c.metadata.get("scope", "")),
            None
        )
        self.assertIsNotNone(method_or_class)

        # Check function chunk
        fn_chunk = next((c for c in chunks if c.metadata.get("symbol_name") == "generate_token"), None)
        self.assertIsNotNone(fn_chunk)
        self.assertIn("@decorator_one", fn_chunk.page_content)
        self.assertIn("def generate_token", fn_chunk.page_content)

        # Verify contextual breadcrumb header
        self.assertTrue(fn_chunk.page_content.startswith("# File: auth/tokens.py"))
        self.assertIn("Lines:", fn_chunk.page_content)

    def test_typescript_structural_chunking(self):
        sample_ts = '''import { useState, useEffect } from "react";

export interface UserProfile {
    id: string;
    name: string;
    email: string;
}

export class UserService {
    private apiEndpoint: string;
    constructor(endpoint: string) {
        this.apiEndpoint = endpoint;
    }
    public async fetchUser(id: string): Promise<UserProfile> {
        return { id, name: "Alice", email: "alice@example.com" };
    }
}

export async function loginUser(email: string): Promise<boolean> {
    console.log("Logging in " + email);
    return true;
}
'''
        doc = Document(page_content=sample_ts, metadata={"source": "src/services/user.ts"})
        chunks = self.chunker.chunk_document(doc)

        self.assertGreater(len(chunks), 1)

        # Verify breadcrumb comment prefix is // for typescript
        self.assertTrue(chunks[0].page_content.startswith("// File: src/services/user.ts"))

        # Verify loginUser is parsed as a declaration
        func_chunk = next((c for c in chunks if "loginUser" in c.metadata.get("scope", "")), None)
        self.assertIsNotNone(func_chunk)
        self.assertIn("export async function loginUser", func_chunk.page_content)

    def test_go_structural_chunking(self):
        sample_go = '''package main

import "fmt"

type ServerConfig struct {
    Host string
    Port int
}

func StartServer(cfg ServerConfig) error {
    fmt.Printf("Starting on %s:%d\\n", cfg.Host, cfg.Port)
    return nil
}
'''
        doc = Document(page_content=sample_go, metadata={"source": "cmd/server/main.go"})
        chunks = self.chunker.chunk_document(doc)

        self.assertGreater(len(chunks), 0)
        start_server = next((c for c in chunks if "StartServer" in c.metadata.get("scope", "")), None)
        self.assertIsNotNone(start_server)
        self.assertIn("func StartServer", start_server.page_content)

    def test_document_chunker_automatic_code_routing(self):
        py_doc = Document(
            page_content="def calculate_sum(a: int, b: int) -> int:\n    return a + b\n",
            metadata={"source": "math/calc.py"}
        )
        txt_doc = Document(
            page_content="This is standard documentation prose explaining the service.",
            metadata={"source": "docs/readme.txt"}
        )

        chunks = DocumentChunker.chunk_documents_recursive([py_doc, txt_doc])
        self.assertEqual(len(chunks), 2)

        py_chunk = next(c for c in chunks if "math/calc.py" in c.metadata.get("source", ""))
        self.assertEqual(py_chunk.metadata.get("language"), "python")
        self.assertIn("# File: math/calc.py", py_chunk.page_content)


class TestCodebaseLoaderAndScale(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(dir=os.getcwd(), prefix=".tmp_codebase_test_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_codebase_with_gitignore_and_ignores(self):
        # Create a realistic directory tree
        os.makedirs(os.path.join(self.temp_dir, "src", "core"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "node_modules", "pkg"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, ".git"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "build"), exist_ok=True)

        # 1. Valid source files
        with open(os.path.join(self.temp_dir, "src", "core", "app.py"), "w") as f:
            f.write("def run():\n    print('Running')\n")
        with open(os.path.join(self.temp_dir, "src", "index.ts"), "w") as f:
            f.write("export const APP_NAME = 'Monster';\n")

        # 2. Ignored files
        with open(os.path.join(self.temp_dir, "node_modules", "pkg", "index.js"), "w") as f:
            f.write("console.log('Ignored');")
        with open(os.path.join(self.temp_dir, ".git", "config"), "w") as f:
            f.write("[core]")
        with open(os.path.join(self.temp_dir, "build", "out.min.js"), "w") as f:
            f.write("var x=1;")

        # 3. .gitignore
        with open(os.path.join(self.temp_dir, ".gitignore"), "w") as f:
            f.write("secret.py\n*.tmp\n")
        with open(os.path.join(self.temp_dir, "secret.py"), "w") as f:
            f.write("SECRET = 'do_not_commit'\n")

        docs = DocumentLoader.load_codebase(self.temp_dir)
        source_paths = [d.metadata["relative_path"] for d in docs]

        # Ensure valid files loaded
        self.assertIn("src/core/app.py", source_paths)
        self.assertIn("src/index.ts", source_paths)

        # Ensure ignored files were excluded
        self.assertNotIn("node_modules/pkg/index.js", source_paths)
        self.assertNotIn(".git/config", source_paths)
        self.assertNotIn("build/out.min.js", source_paths)
        self.assertNotIn("secret.py", source_paths)

    def test_scale_large_codebase_throughput(self):
        """
        Simulate a multi-module repository with ~10,000 lines of code across 50 files
        and verify high-throughput ingestion and logical chunk generation.
        """
        import time
        os.makedirs(os.path.join(self.temp_dir, "modules"), exist_ok=True)

        function_template = """
def process_event_{idx}_{fn}(event_id: str, payload: dict) -> dict:
    \"\"\"Handles event processing for event {idx}-{fn}.\"\"\"
    status = "success"
    result = {{
        "event_id": event_id,
        "processed_at": 1700000000,
        "status": status,
        "step": {fn}
    }}
    return result
"""
        total_files = 30
        for i in range(total_files):
            file_path = os.path.join(self.temp_dir, "modules", f"service_{i}.py")
            with open(file_path, "w") as f:
                f.write(f'"""Service module {i}."""\nimport os\n\n')
                for fn in range(15):
                    f.write(function_template.format(idx=i, fn=fn))

        # Measure loading and chunking speed
        t0 = time.perf_counter()
        docs = DocumentLoader.load_codebase(self.temp_dir)
        t_load = time.perf_counter() - t0

        t1 = time.perf_counter()
        chunks = DocumentChunker.chunk_documents_recursive(docs)
        t_chunk = time.perf_counter() - t1

        total_lines = sum(d.metadata.get("line_count", 0) for d in docs)
        self.assertEqual(len(docs), total_files)
        self.assertGreater(total_lines, 5000)
        self.assertGreater(len(chunks), total_files * 10)

        # Throughput check: loading and chunking should be under 1.5 seconds
        self.assertLess(t_load + t_chunk, 2.0)


if __name__ == "__main__":
    unittest.main()
