"""
core/faithfulness_verifier.py
─────────────────────────────
Post-generation hallucination guard.

Sends the question, the generated answer, and the retrieved context to
the LLM and asks it to judge whether every claim in the answer is
grounded in the context.

Returns a faithfulness score [0, 1] and a list of hallucinated claims.
If verification errors or score < min_score, it flags faithful=False so
the pipeline can return a safe escalation response.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

_VERIFIER_PROMPT = """\
You are a faithfulness checker for a RAG (Retrieval-Augmented Generation) system.
Given a question, a generated answer, and the source context, check whether the
answer is fully grounded in the context.

Respond ONLY with a JSON object in this exact format, nothing else:
{
  "faithful": true,
  "score": 0.92,
  "hallucinated_claims": [],
  "reason": "one line explanation"
}

Scoring guide:
- 1.0  : every claim is directly supported by the context
- 0.7+ : mostly supported, minor inferences acceptable
- 0.4–0.7: some claims not in context
- < 0.4: significant hallucination

"faithful" should be true when score >= 0.6.
"hallucinated_claims": list the specific sentences / phrases that are NOT in context.
"""

_MAX_CONTEXT_CHARS = 3000


def verify(
    query: str,
    answer: str,
    context: str,
    llm=None,
    min_score: float = 0.6,
) -> dict:
    """
    Verify that the generated answer is grounded in the retrieved context.

    If the verifier errors, it returns faithful=False with score=0.0
    to prevent unverified or hallucinated responses from reaching users.
    """
    if llm is None:
        try:
            from core.llm import get_llm
            llm = get_llm()
        except Exception as exc:
            logger.warning("faithfulness_verifier_llm_unavailable", extra={"error": str(exc)})
            return _fallback(query, answer, f"LLM unavailable: {exc}")

    truncated_context = context[:_MAX_CONTEXT_CHARS]
    if len(context) > _MAX_CONTEXT_CHARS:
        truncated_context += "\n[... context truncated ...]"

    user_message = (
        f"Question: {query}\n\n"
        f"Generated Answer: {answer}\n\n"
        f"Source Context:\n{truncated_context}\n\n"
        "Check if the answer is fully supported by the context:"
    )

    prompt = f"{_VERIFIER_PROMPT}\n\n{user_message}"
    try:
        raw = str(llm.invoke(prompt)).strip()
        json_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
        else:
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
            result = json.loads(raw.strip())
    except Exception as exc:
        logger.warning("faithfulness_verifier_error", extra={"error": str(exc)})
        return _fallback(query, answer, f"Verification failed to complete: {exc}")

    # Normalise output
    score = float(result.get("score", 0.0))
    is_faithful = bool(result.get("faithful", False)) and (score >= min_score)
    result["faithful"] = is_faithful
    result["score"] = score
    result.setdefault("hallucinated_claims", [])
    result.setdefault("reason", "")
    result["query"] = query
    result["answer"] = answer
    result["error"] = False

    if not is_faithful:
        logger.warning(
            "faithfulness_low_score",
            extra={
                "score": result.get("score"),
                "hallucinated_claims": result.get("hallucinated_claims"),
                "query": query[:100],
            },
        )

    return result


def _fallback(query: str, answer: str, reason: str = "verifier unavailable or error") -> dict:
    """Safe fallback: unverified answers are treated as unfaithful."""
    return {
        "faithful": False,
        "score": 0.0,
        "hallucinated_claims": [reason],
        "reason": f"Verification error: {reason}",
        "query": query,
        "answer": answer,
        "error": True,
    }
