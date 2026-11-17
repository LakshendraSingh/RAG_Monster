"""
core/query_router.py
────────────────────
Classifies incoming queries into one of four semantic types and maps
each type to the best retrieval strategy and top-k value.

QueryTypes:
  factual      – simple fact lookup           → dense,  top_k=3
  reasoning    – multi-step / explanation     → hybrid, top_k=5
  comparative  – comparing ≥2 things          → hybrid, top_k=6
  ambiguous    – vague / needs clarification  → sparse, top_k=3

The classifier uses the RAG_MONSTER's own LLM registry (get_llm) so
it automatically benefits from whatever model the user has configured —
no separate Ollama dependency.

Ported and adapted from self-adapting-rag-main/retrieval/query_analyzer.py
and retrieval/strategy_router.py.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

logger = logging.getLogger(__name__)

QueryType = Literal["factual", "reasoning", "comparative", "ambiguous"]
Strategy  = Literal["dense", "sparse", "hybrid"]

# Default routing table — may be overridden at runtime by AdaptationEngine
DEFAULT_STRATEGY_MAP: dict[str, dict] = {
    "factual":     {"strategy": "dense",  "top_k": 3},
    "reasoning":   {"strategy": "hybrid", "top_k": 5},
    "comparative": {"strategy": "hybrid", "top_k": 6},
    "ambiguous":   {"strategy": "sparse", "top_k": 3},
}

_CLASSIFIER_PROMPT = """\
You are a query classifier for a RAG (Retrieval-Augmented Generation) system.
Classify the user query into exactly one of these types:

- factual:     Simple fact lookup. A single, direct answer exists.
- reasoning:   Requires multi-step thinking, explanation, or analysis.
- comparative: Compares or contrasts two or more things.
- ambiguous:   Too vague, unclear, or requires clarification.

Respond ONLY with a JSON object in this exact format, nothing else:
{"type": "factual", "confidence": 0.95, "reasoning": "one line explanation"}
"""


def classify_query(query: str, llm=None) -> dict:
    """
    Classify a query using the configured LLM.
    Falls back to 'factual' if classification fails.

    Args:
        query: The user's question.
        llm: An optional LangChain-compatible LLM instance. If not supplied,
             a lightweight model is obtained from the registry.

    Returns:
        dict with keys: type, confidence, reasoning, query
    """
    if llm is None:
        try:
            from core.llm import get_llm
            llm = get_llm()
        except Exception as exc:
            logger.warning("query_router_llm_unavailable", extra={"error": str(exc)})
            return _fallback("factual", "LLM unavailable", query)

    prompt = f"{_CLASSIFIER_PROMPT}\n\nClassify this query: {query}"
    try:
        raw = llm.invoke(prompt)
        raw = str(raw).strip()

        # parse json robustly, extracting JSON substring or stripping markdown fences
        json_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
        else:
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
            result = json.loads(raw.strip())
    except json.JSONDecodeError:
        logger.warning("query_router_parse_error", extra={"raw": raw[:200]})
        result = _fallback("factual", "JSON parse failed", query)
    except Exception as exc:
        logger.warning("query_router_error", extra={"error": str(exc)})
        result = _fallback("factual", str(exc), query)

    valid_types = {"factual", "reasoning", "comparative", "ambiguous"}
    if result.get("type") not in valid_types:
        result["type"] = "factual"

    result["query"] = query
    return result


def route(query: str, strategy_map: dict | None = None, llm=None) -> dict:
    """
    Classify query and return the retrieval routing decision.

    Returns:
        dict with keys: query, type, strategy, top_k, confidence, reasoning
    """
    smap = strategy_map or DEFAULT_STRATEGY_MAP
    classification = classify_query(query, llm=llm)
    qtype = classification["type"]
    config = smap.get(qtype, smap.get("factual", {"strategy": "dense", "top_k": 5}))

    return {
        "query":      query,
        "type":       qtype,
        "strategy":   config["strategy"],
        "top_k":      config["top_k"],
        "confidence": classification.get("confidence", 0.5),
        "reasoning":  classification.get("reasoning", ""),
    }


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _fallback(qtype: str, reason: str, query: str) -> dict:
    return {
        "type":       qtype,
        "confidence": 0.0,
        "reasoning":  reason,
        "query":      query,
    }
