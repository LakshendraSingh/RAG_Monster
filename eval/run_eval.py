"""
Evaluation harness for RAG_MONSTER.

Reports:
  - recall@5 (answerable queries)
  - MRR (Mean Reciprocal Rank, answerable queries)
  - answer_correctness (keyword/fact match on answerable queries)
  - citation_accuracy (valid [n] citations pointing to expected source)
  - abstention_rate (on unanswerable queries)
  - latency_avg_s / latency_p95_s
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Use isolated eval Chroma DB so we don't touch user's production ./chroma_db
os.environ.setdefault("CHROMA_PATH", str(ROOT_DIR / "eval_chroma_db"))

from main_rag import RAGPipeline  # noqa: E402


ABSTENTION_PATTERNS = [
    "don't have enough information",
    "do not have enough information",
    "not enough information",
    "cannot be found",
    "can't be found",
    "not mentioned",
    "not provided in",
    "not in the context",
    "not in the provided",
    "no information",
    "does not contain",
    "doesn't contain",
    "not specified",
    "unable to find",
    "cannot answer",
    "i don't know",
]


def load_dataset(path: Path) -> list[dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def check_abstention(answer: str, result: dict[str, Any]) -> bool:
    if result.get("abstained"):
        return True
    lower = answer.lower()
    return any(p in lower for p in ABSTENTION_PATTERNS)


def check_answer_correctness(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in lower)
    return hits / len(expected_keywords)


def check_citation_accuracy(
    answer: str,
    sources: list[dict[str, Any]],
    retrieved_sources: list[str],
    expected_source: str,
) -> float:
    """Check if answer contains valid [n] citations and cites the expected source."""
    cited_nums = [int(m) for m in re.findall(r"\[(\d+)\]", answer)]
    if not cited_nums:
        return 0.0

    # Determine source list from structured sources (Phase 2+) or retrieved_sources
    src_names = [
        os.path.basename(s.get("source", "") if isinstance(s, dict) else str(s))
        for s in sources
    ] if sources else retrieved_sources
    if not src_names:
        return 0.0

    # All cited indices must be in valid range 1..len(src_names)
    if any(n < 1 or n > len(src_names) for n in cited_nums):
        return 0.0

    # At least one cited source must match expected_source
    cited_sources = {src_names[n - 1] for n in cited_nums}
    return 1.0 if expected_source in cited_sources else 0.0


def retrieve_top5_sources(pipeline: RAGPipeline, question: str) -> list[str]:
    """Retrieve top-5 source filenames cleanly for both baseline and Phase 1+."""
    retriever = pipeline.retriever
    if hasattr(retriever, "retrieve_with_scores"):
        docs_and_scores = retriever.retrieve_with_scores(question, top_k=5)
        return [os.path.basename(d.metadata.get("source", "")) for d, _ in docs_and_scores]
    if hasattr(retriever, "retrieve"):
        docs = retriever.retrieve(question, top_k=5)
        return [os.path.basename(d.metadata.get("source", "")) for d in docs]

    # Baseline (Phase 0) retriever
    docs = retriever.base_retriever.invoke(question)
    if retriever.use_reranker and docs:
        docs = retriever.reranker.rerank(question, docs, 5)
    else:
        docs = docs[:5]
    return [os.path.basename(d.metadata.get("source", "")) for d in docs]


def run_eval(
    dataset_path: Path,
    data_dir: Path,
    output_path: Path,
    skip_llm: bool = False,
    max_llm_queries: int | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    eval_chroma = Path(os.environ["CHROMA_PATH"])
    if eval_chroma.exists():
        shutil.rmtree(eval_chroma)

    print(f"Initializing RAGPipeline (CHROMA_PATH={eval_chroma})...")
    pipeline = RAGPipeline(use_reranker=True, model=model)

    # Index sample docs
    doc_files = sorted(p for p in data_dir.iterdir() if p.is_file())
    print(f"Indexing {len(doc_files)} evaluation documents...")
    for doc_path in doc_files:
        pipeline.index_document(str(doc_path))

    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} evaluation queries.")

    recall_hits: list[float] = []
    rr_scores: list[float] = []
    correctness_scores: list[float] = []
    citation_scores: list[float] = []
    abstention_hits: list[float] = []
    latencies: list[float] = []
    per_item_results: list[dict[str, Any]] = []

    llm_calls = 0
    llm_errors = 0

    # Determine which items get LLM evaluation (stratified if max_llm_queries is set)
    if skip_llm:
        llm_ids: set[str] = set()
    elif max_llm_queries is not None and max_llm_queries < len(dataset):
        ans_items = [x for x in dataset if x.get("answerable", True)]
        unans_items = [x for x in dataset if not x.get("answerable", True)]
        n_unans = max(1, round(max_llm_queries * len(unans_items) / len(dataset)))
        n_ans = max(1, max_llm_queries - n_unans)
        # Ensure multi-turn item (last answerable, q24) is included if available
        selected_ans = ans_items[: max(0, n_ans - 1)] + ([ans_items[-1]] if ans_items and n_ans > 0 else [])
        selected_unans = unans_items[:n_unans]
        llm_ids = {x["id"] for x in selected_ans + selected_unans}
    else:
        llm_ids = {x["id"] for x in dataset}

    for item in dataset:
        qid = item["id"]
        question = item["question"]
        expected_source = item.get("source_doc", "")
        answerable = item.get("answerable", True)
        expected_keywords = item.get("expected_keywords", [])
        chat_history = item.get("chat_history", [])

        session_id = f"eval_{qid}"
        pipeline.cache.clear_chat_history(session_id)
        for turn in chat_history:
            pipeline.cache.add_chat_message(session_id, turn["role"], turn["content"])

        # 1. Measure retrieval
        t0 = time.perf_counter()
        retrieved_sources = retrieve_top5_sources(pipeline, question)
        retrieval_time = time.perf_counter() - t0

        if answerable and expected_source:
            hit = 1.0 if expected_source in retrieved_sources else 0.0
            recall_hits.append(hit)
            rr = 0.0
            for idx, src in enumerate(retrieved_sources, start=1):
                if src == expected_source:
                    rr = 1.0 / idx
                    break
            rr_scores.append(rr)
        else:
            hit = None
            rr = None

        # 2. Measure full query (retrieval + generation)
        answer = ""
        sources: list[dict[str, Any]] = []
        abstained = None
        correctness = None
        cit_acc = None
        total_latency = retrieval_time

        run_this_llm = qid in llm_ids
        if run_this_llm:
            t_q0 = time.perf_counter()
            try:
                result = pipeline.query(session_id, question, model=model)
                total_latency = time.perf_counter() - t_q0
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                llm_calls += 1

                if answerable:
                    correctness = check_answer_correctness(answer, expected_keywords)
                    correctness_scores.append(correctness)
                    cit_acc = check_citation_accuracy(answer, sources, retrieved_sources, expected_source)
                    citation_scores.append(cit_acc)
                else:
                    abstained = check_abstention(answer, result)
                    abstention_hits.append(1.0 if abstained else 0.0)
            except Exception as e:
                total_latency = time.perf_counter() - t_q0
                llm_errors += 1
                answer = f"[LLM_ERROR: {e}]"
                print(f"  [{qid}] LLM error: {e}")

        latencies.append(total_latency)
        per_item_results.append({
            "id": qid,
            "answerable": answerable,
            "expected_source": expected_source,
            "retrieved_sources": retrieved_sources,
            "recall_at_5": hit,
            "reciprocal_rank": rr,
            "answer_correctness": correctness,
            "citation_accuracy": cit_acc,
            "abstained": abstained,
            "latency_s": round(total_latency, 3),
            "answer_preview": answer[:200],
        })

    def _mean(vals: list[float]) -> float | None:
        return round(sum(vals) / len(vals), 4) if vals else None

    sorted_lat = sorted(latencies)
    p95_idx = min(len(sorted_lat) - 1, int(0.95 * len(sorted_lat))) if sorted_lat else 0

    summary = {
        "total_queries": len(dataset),
        "answerable_queries": len(recall_hits),
        "unanswerable_queries": sum(1 for x in dataset if not x.get("answerable", True)),
        "llm_calls_attempted": llm_calls,
        "llm_errors": llm_errors,
        "metrics": {
            "recall_at_5": _mean(recall_hits),
            "mrr": _mean(rr_scores),
            "answer_correctness": _mean(correctness_scores),
            "citation_accuracy": _mean(citation_scores),
            "abstention_rate": _mean(abstention_hits),
            "latency_avg_s": round(statistics.mean(latencies), 3) if latencies else None,
            "latency_p95_s": round(sorted_lat[p95_idx], 3) if sorted_lat else None,
        },
        "items": per_item_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EVAL SUMMARY ===")
    for k, v in summary["metrics"].items():
        print(f"  {k:22s}: {v}")
    print(f"Saved results to {output_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG_MONSTER evaluation harness.")
    parser.add_argument("--dataset", default=str(ROOT_DIR / "eval" / "dataset.jsonl"))
    parser.add_argument("--data-dir", default=str(ROOT_DIR / "eval" / "data"))
    parser.add_argument("--output", default=str(ROOT_DIR / "eval" / "results_baseline.json"))
    parser.add_argument("--skip-llm", action="store_true", help="Only evaluate retrieval + latency.")
    parser.add_argument("--max-llm-queries", type=int, default=None, help="Cap LLM calls (for rate-limited keys).")
    parser.add_argument("--model", default=None, help="Override LLM model.")
    args = parser.parse_args()

    run_eval(
        dataset_path=Path(args.dataset),
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        skip_llm=args.skip_llm,
        max_llm_queries=args.max_llm_queries,
        model=args.model,
    )


if __name__ == "__main__":
    main()
