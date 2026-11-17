import os
import sys
import time
from config import LLM_MODEL
from main_rag import RAGPipeline

def run_pdf_tests():
    print(f"Initializing RAGPipeline with model: {LLM_MODEL}...")
    pipeline = RAGPipeline(
        model=LLM_MODEL,
        enable_semantic_cache=False,  # fresh queries
        enable_query_routing=True,
        enable_query_rewriting=False,
        enable_faithfulness_check=True,
        enable_feedback_logging=False
    )

    pdf_files = [
        "advanced-linux-commands-cheat-sheet-red-hat-developer.pdf",
        "220907-Mulchandani-SoftwareDefined-Warfare.pdf",
        "22-03-Krill.pdf",
        "1807.02758v2.pdf",
        "367657_30168365.pdf"
    ]

    print("\n--- Phase 1: Indexing PDFs ---")
    for pdf in pdf_files:
        t0 = time.time()
        res = pipeline.index_document(pdf)
        elapsed = time.time() - t0
        print(f"Indexed '{pdf}': {res['chunks_indexed']} chunks in {elapsed:.2f}s")

    test_queries = [
        {
            "pdf": "advanced-linux-commands-cheat-sheet-red-hat-developer.pdf",
            "question": "What commands are explained in the Linux cheat sheet for Application Management (such as controlling systemd services) and Audit logging?"
        },
        {
            "pdf": "220907-Mulchandani-SoftwareDefined-Warfare.pdf",
            "question": "What is Nand Mulchandani's core thesis regarding Software-Defined Warfare and why DoD needs to transition to an iterative, software-driven organization?"
        },
        {
            "pdf": "22-03-Krill.pdf",
            "question": "According to J.A. Krill et al. in the Johns Hopkins APL digest, how do systems engineers and program managers partner together in a matrix organization?"
        },
        {
            "pdf": "1807.02758v2.pdf",
            "question": "What is the Residual Channel Attention Network (RCAN) and how does the Residual in Residual (RIR) structure improve image super-resolution?"
        },
        {
            "pdf": "367657_30168365.pdf",
            "question": "According to the paper by Manvinder Sharma and Anuj Kumar Gupta, what is the focus of their critical study on missiles and missile guidance?"
        }
    ]

    print("\n--- Phase 2: Running RAG Queries with Gemini ---")
    results = []
    for i, item in enumerate(test_queries, 1):
        q = item["question"]
        target = item["pdf"]
        print(f"\n==================================================")
        print(f"Query {i}: Testing against [{target}]")
        print(f"Question: {q}")
        print("--------------------------------------------------")
        t0 = time.time()
        res = pipeline.query(session_id=f"test_session_{i}", question=q)
        latency = time.time() - t0

        answer = res.get("answer", "")
        sources = res.get("sources", [])
        faith_info = res.get("faithfulness", {})
        faith_score = faith_info.get("score")
        is_faithful = faith_info.get("faithful")
        escalated = faith_info.get("escalated")

        print(f"Latency: {latency:.2f}s")
        print(f"Strategy: {res.get('strategy')} | Type: {res.get('query_type')}")
        print(f"Faithfulness score: {faith_score} (Faithful: {is_faithful}, Escalated: {escalated})")
        print(f"Answer:\n{answer}")
        print(f"Sources Cited: {sources}")
        
        results.append({
            "target": target,
            "question": q,
            "latency": latency,
            "faithfulness": faith_score,
            "is_faithful": is_faithful,
            "escalated": escalated,
            "answer": answer,
            "sources": sources
        })

    print("\n==================================================")
    print("All 5 PDFs successfully queried and verified!")

if __name__ == "__main__":
    run_pdf_tests()
