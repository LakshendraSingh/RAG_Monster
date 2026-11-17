"""
core/adaptation_engine.py
─────────────────────────
Self-adapting retrieval strategy selector.

After enough query logs have accumulated, the engine compares how well
each retrieval strategy (dense / sparse / hybrid) has been performing
for each query type (factual / reasoning / comparative / ambiguous).

It uses a composite score:
    score = 0.4 × avg_relevance + 0.4 × avg_faithfulness + 0.2 × avg_user_feedback

If a strategy outperforms the current default for a given query type
(and has at least MIN_SAMPLES observations) it is promoted automatically.

The updated strategy_map is handed back to RAGPipeline and used by
QueryRouter from that point on.

Ported and adapted from self-adapting-rag-main/adaptation/adaptation_engine.py.
"""

from __future__ import annotations

import logging
from typing import Optional

from core.feedback_logger import FeedbackLogger
from core.query_router import DEFAULT_STRATEGY_MAP

logger = logging.getLogger(__name__)

# Minimum samples before we trust the data for a given type:strategy pair
MIN_SAMPLES = 5

SCORE_WEIGHTS = {
    "avg_relevance":    0.4,
    "avg_faithfulness": 0.4,
    "avg_feedback":     0.2,
}


def _composite_score(stats: dict) -> float:
    return (
        stats["avg_relevance"]    * SCORE_WEIGHTS["avg_relevance"]
        + stats["avg_faithfulness"] * SCORE_WEIGHTS["avg_faithfulness"]
        + stats["avg_feedback"]     * SCORE_WEIGHTS["avg_feedback"]
    )


class AdaptationEngine:
    """
    Periodically recalibrates the query-type → retrieval-strategy mapping
    based on observed performance in the feedback log.

    Usage:
        engine = AdaptationEngine(feedback_logger)
        result = engine.run()   # call this on a schedule (e.g. every hour)
        pipeline.set_strategy_map(result["updated_map"])
    """

    def __init__(self, feedback_logger: Optional[FeedbackLogger] = None):
        self.logger = feedback_logger or FeedbackLogger()
        # Start with the hardcoded defaults; adapt from there
        self.strategy_map: dict = {k: v.copy() for k, v in DEFAULT_STRATEGY_MAP.items()}

    def run(self) -> dict:
        """
        Run one adaptation cycle.

        Returns:
            dict with: updated_map, changes, stats, message
        """
        stats = self.logger.get_strategy_stats()

        if not stats:
            return {
                "updated_map": self.strategy_map,
                "changes":     [],
                "stats":       {},
                "message":     "No data yet — need more query logs",
            }

        new_map, changes = self._adapt(stats)
        self.strategy_map = new_map

        message = (
            f"{len(changes)} strategy update(s) applied"
            if changes else "No changes — current strategies are optimal"
        )
        logger.info("adaptation_engine_run", extra={"changes": changes, "stats": stats})

        return {
            "updated_map": new_map,
            "changes":     changes,
            "stats":       stats,
            "message":     message,
        }

    def get_strategy(self, query_type: str) -> dict:
        """Return the current routing config for a query type."""
        return self.strategy_map.get(
            query_type,
            DEFAULT_STRATEGY_MAP.get("factual", {"strategy": "dense", "top_k": 5}),
        )

    # ──────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────

    def _adapt(self, stats: dict) -> tuple[dict, list]:
        new_map = {k: v.copy() for k, v in self.strategy_map.items()}
        changes = []

        # Group stats by query type
        by_type: dict[str, dict] = {}
        for key, val in stats.items():
            qtype, strategy = key.split(":", 1)
            by_type.setdefault(qtype, {})[strategy] = val

        for qtype, strategies in by_type.items():
            if qtype not in new_map:
                continue

            best_strategy = None
            best_score    = -1.0

            for strategy, stat in strategies.items():
                if stat["count"] < MIN_SAMPLES:
                    continue
                score = _composite_score(stat)
                if score > best_score:
                    best_score    = score
                    best_strategy = strategy

            current_strategy = new_map[qtype]["strategy"]
            if best_strategy and best_strategy != current_strategy:
                new_map[qtype]["strategy"] = best_strategy
                changes.append({
                    "query_type":   qtype,
                    "old_strategy": current_strategy,
                    "new_strategy": best_strategy,
                    "score":        round(best_score, 3),
                })
                logger.info(
                    "adaptation_strategy_updated",
                    extra={
                        "query_type":   qtype,
                        "old":          current_strategy,
                        "new":          best_strategy,
                        "score":        round(best_score, 3),
                    },
                )

        return new_map, changes
