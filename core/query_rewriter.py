"""
core/query_rewriter.py
──────────────────────
Rewrites a query when retrieval quality is poor, then optionally retries.

Logic:
  1. Score retrieved chunks (avg relevance via CrossEncoder scores or
     relevance_score field if already scored).
  2. If top score < threshold → ask the LLM to rephrase the query.
  3. Return (rewritten_query, should_retry) so the caller can re-retrieve.

Ported and adapted from self-adapting-rag-main/retrieval/query_rewriter.py.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_REWRITER_PROMPT = """\
You are a query rewriting assistant for a RAG (Retrieval-Augmented Generation) system.
The original query failed to retrieve sufficiently relevant results.
Rewrite it to be more specific, use different keywords, or decompose it into a clearer question.

Respond ONLY with a JSON object in this exact format, nothing else:
{"rewritten": "the new query", "strategy": "one line on what you changed"}
"""


def rewrite_query(query: str, reason: str = "", llm=None) -> dict:
    """
    Ask the LLM to rephrase a low-quality query.

    Returns:
        dict with: original, rewritten, strategy
    """
    if llm is None:
        try:
            from core.llm import get_llm
            llm = get_llm()
        except Exception as exc:
            logger.warning("rewriter_llm_unavailable", extra={"error": str(exc)})
            return {"original": query, "rewritten": query, "strategy": "LLM unavailable"}

    user_msg = f"Original query: {query}"
    if reason:
        user_msg += f"\nReason retrieval failed: {reason}"

    prompt = f"{_REWRITER_PROMPT}\n\n{user_msg}"
    try:
        raw = str(llm.invoke(prompt)).strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
        result = json.loads(raw.strip())
    except Exception as exc:
        logger.warning("rewriter_parse_error", extra={"error": str(exc)})
        result = {"rewritten": query, "strategy": "parse failed — using original"}

    result["original"] = query
    return result


def rewrite_if_needed(
    query: str,
    docs: list[Any],
    threshold: float = 0.5,
    llm=None,
) -> dict:
    """
    Evaluate retrieval quality and rewrite the query only if needed.

    Args:
        query:     The original query string.
        docs:      Retrieved LangChain Document objects.
                   Relevance scores are read from doc.metadata["relevance_score"]
                   if present (set by the reranker); otherwise rewriting is
                   triggered only when ``docs`` is empty.
        threshold: If the top relevance score is below this value, rewrite.
        llm:       Optional LLM override.

    Returns:
        dict with: should_retry, original, rewritten, strategy, avg_score, top_score
    """
    if not docs:
        result = rewrite_query(query, reason="no chunks retrieved at all", llm=llm)
        return {"should_retry": True, **result, "avg_score": 0.0, "top_score": 0.0}

    # Extract scores if the reranker attached them
    scores = [
        float(doc.metadata.get("relevance_score", 0.5))
        for doc in docs
    ]
    avg_score = sum(scores) / len(scores)
    top_score = max(scores)

    if top_score < threshold:
        reason = f"best chunk score was only {top_score:.2f}, below threshold {threshold}"
        logger.info("query_rewriter_triggered", extra={"reason": reason, "query": query})
        result = rewrite_query(query, reason=reason, llm=llm)
        return {
            "should_retry": True,
            "avg_score":    avg_score,
            "top_score":    top_score,
            **result,
        }

    return {
        "should_retry": False,
        "original":     query,
        "rewritten":    query,
        "strategy":     "no rewrite needed",
        "avg_score":    avg_score,
        "top_score":    top_score,
    }
