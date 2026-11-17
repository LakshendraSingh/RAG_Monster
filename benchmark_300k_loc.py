"""
benchmark_300k_loc.py
─────────────────────
Stress-test and benchmark for indexing and logically chunking a 300,000+ Lines of Code (LoC)
codebase with AST parsing, structural syntax chunking, and BM25 indexation.
"""

import os
import shutil
import tempfile
import time
from core.document_loader import DocumentLoader
from core.chunker import DocumentChunker
from core.retriever import RAGRetriever

def generate_synthetic_300k_codebase(target_dir: str, target_loc: int = 300_000):
    print(f"Generating synthetic codebase with ~{target_loc:,} lines of code...")
    t0 = time.perf_counter()

    # Create directory structure
    langs = ["python", "typescript", "go", "rust"]
    for lang in langs:
        os.makedirs(os.path.join(target_dir, "services", lang), exist_ok=True)
        os.makedirs(os.path.join(target_dir, "pkg", lang), exist_ok=True)

    # 1. Python file template (~500 lines per file)
    py_func = """
    @metrics.timer("process_item")
    def execute_transaction_{fn}(self, tx_id: str, amount: float, account_from: str, account_to: str) -> dict:
        \"\"\"Process financial transaction {fn} with validation.\"\"\"
        if amount <= 0:
            raise ValueError("Amount must be positive")
        tx_record = {{
            "tx_id": tx_id,
            "from": account_from,
            "to": account_to,
            "amount": amount,
            "timestamp": 1700000000 + {fn},
            "status": "COMPLETED",
            "audit_hash": hashlib.sha256(f"{{tx_id}}_{{amount}}".encode()).hexdigest(),
        }}
        self.ledger.append(tx_record)
        return tx_record
"""
    # 2. TypeScript template (~400 lines per file)
    ts_func = """
export interface PaymentRecord_{fn} {{
    id: string;
    amount: number;
    sender: string;
    receiver: string;
    status: "pending" | "settled" | "failed";
}}

export async function processPayment_{fn}(record: PaymentRecord_{fn}): Promise<boolean> {{
    console.log("Processing payment: " + record.id);
    if (record.amount < 0) return false;
    return true;
}}
"""
    # 3. Go template (~300 lines per file)
    go_func = """
type WorkerPool_{fn} struct {{
    ID       string
    Capacity int
    Active   bool
}}

func (w *WorkerPool_{fn}) DispatchJob(jobID string, payload []byte) error {{
    if !w.Active {{
        return fmt.Errorf("worker pool %s is inactive", w.ID)
    }}
    return nil
}}
"""

    current_loc = 0
    file_count = 0

    while current_loc < target_loc:
        file_count += 1
        # Alternate between Python, TypeScript, and Go
        mod = file_count % 3
        if mod == 0:
            fpath = os.path.join(target_dir, "services", "python", f"ledger_service_{file_count}.py")
            content = '"""Financial Ledger Service Module."""\nimport hashlib\nimport os\n\nclass LedgerManager:\n    def __init__(self):\n        self.ledger = []\n'
            for fn in range(25):
                content += py_func.format(fn=fn)
        elif mod == 1:
            fpath = os.path.join(target_dir, "services", "typescript", f"payment_gateway_{file_count}.ts")
            content = 'import { Logger } from "@core/logger";\n\n'
            for fn in range(20):
                content += ts_func.format(fn=fn)
        else:
            fpath = os.path.join(target_dir, "pkg", "go", f"worker_pool_{file_count}.go")
            content = 'package pool\n\nimport "fmt"\n\n'
            for fn in range(20):
                content += go_func.format(fn=fn)

        lines = content.count("\n")
        current_loc += lines

        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)

    elapsed = time.perf_counter() - t0
    print(f"Generated {current_loc:,} LoC across {file_count} files in {elapsed:.2f}s.\n")
    return current_loc, file_count


def benchmark_300k():
    temp_dir = tempfile.mkdtemp(dir=os.getcwd(), prefix=".benchmark_300k_")
    try:
        actual_loc, file_count = generate_synthetic_300k_codebase(temp_dir, target_loc=300_000)

        # ── Phase 1: Parallel Ingestion & Traversal ──────────────────────
        print("--- Phase 1: Ingestion & Traversal ---")
        t0 = time.perf_counter()
        docs = DocumentLoader.load_codebase(temp_dir, max_workers=8)
        t_ingest = time.perf_counter() - t0
        ingest_throughput = actual_loc / max(t_ingest, 0.001)
        print(f"Loaded {len(docs)} files ({actual_loc:,} LoC) in {t_ingest:.2f}s")
        print(f"Ingestion Throughput: {ingest_throughput:,.0f} LoC / second")

        # ── Phase 2: AST & Structural Logical Chunking ───────────────────
        print("\n--- Phase 2: AST & Structural Logical Chunking ---")
        t0 = time.perf_counter()
        chunks = DocumentChunker.chunk_documents_recursive(docs, chunk_size=1500)
        t_chunk = time.perf_counter() - t0
        chunk_throughput = actual_loc / max(t_chunk, 0.001)
        print(f"Generated {len(chunks):,} logical chunks in {t_chunk:.2f}s")
        print(f"Chunking Throughput: {chunk_throughput:,.0f} LoC / second")

        # Inspect sample chunks
        print("\n--- Sample Logical Chunks ---")
        py_sample = next(c for c in chunks if c.metadata.get("language") == "python" and c.metadata.get("chunk_type") == "method")
        print(f"[Python Method Chunk]:\n{py_sample.page_content[:300]}...\n")
        ts_sample = next(c for c in chunks if c.metadata.get("language") == "typescript")
        print(f"[TypeScript Chunk]:\n{ts_sample.page_content[:300]}...\n")

        # ── Phase 3: Indexing & Retrieval Verification ───────────────────
        print("--- Phase 3: BM25 Indexing & Search Benchmark ---")
        retriever = RAGRetriever(use_reranker=False)
        t0 = time.perf_counter()
        retriever.add_documents(chunks)
        t_index = time.perf_counter() - t0
        print(f"Indexed {len(chunks):,} chunks into BM25 in {t_index:.2f}s")

        # Search test
        search_query = "execute_transaction_12 audit_hash validation"
        t0 = time.perf_counter()
        results = retriever.retrieve(search_query, top_k=3, strategy="sparse")
        t_search = time.perf_counter() - t0
        print(f"\nSearch Query: '{search_query}' (Latency: {t_search * 1000:.2f}ms)")
        print(f"Top Result:\n{results[0].page_content[:250]}...\n")

        print("==================================================")
        print("BENCHMARK SUMMARY FOR 300k+ LoC CODEBASE:")
        print(f"Total Lines of Code:   {actual_loc:,}")
        print(f"Total Source Files:    {file_count}")
        print(f"Total Logical Chunks:  {len(chunks):,}")
        print(f"Total Ingest & Chunk:  {t_ingest + t_chunk:.2f}s ({(actual_loc / (t_ingest + t_chunk)):,.0f} LoC/sec)")
        print(f"BM25 Retrieval Speed:  {t_search * 1000:.2f}ms")
        print("==================================================")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    benchmark_300k()
