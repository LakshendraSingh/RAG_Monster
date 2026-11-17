"""
core/feedback_logger.py
───────────────────────
SQLite-backed audit log for every RAG query interaction.

Records:
  - query text, type, strategy chosen, top_k
  - retrieval quality (chunks retrieved / passed, avg + top relevance score)
  - faithfulness score of the generated answer
  - whether the query was rewritten and what it was rewritten to
  - final answer text
  - optional user feedback (1 = thumbs-up, 0 = thumbs-down)

The data is used by AdaptationEngine to identify which retrieval
strategies perform best per query type.

Ported and adapted from self-adapting-rag-main/adaptation/feedback_logger.py.
Uses SQLite by default (zero extra infrastructure); Postgres is used when
DATABASE_URL is set in the environment.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from sqlalchemy import (
        Column,
        DateTime,
        Float,
        Integer,
        String,
        Boolean,
        Text,
        create_engine,
        func,
    )
    from sqlalchemy.orm import DeclarativeBase, Session
    _HAS_SQLALCHEMY = True
except ImportError:
    _HAS_SQLALCHEMY = False

_DEFAULT_DB = "sqlite:///./rag_feedback.db"


# ──────────────────────────────────────────────
# ORM Model (only defined when SQLAlchemy present)
# ──────────────────────────────────────────────

if _HAS_SQLALCHEMY:
    class _Base(DeclarativeBase):
        pass

    class QueryLog(_Base):
        __tablename__ = "query_logs"

        id                   = Column(Integer, primary_key=True, autoincrement=True)
        timestamp            = Column(DateTime, default=lambda: datetime.now(timezone.utc))
        query                = Column(Text, nullable=False)
        query_type           = Column(String(32))
        strategy_used        = Column(String(32))
        top_k                = Column(Integer)
        num_chunks_retrieved = Column(Integer)
        num_chunks_passed    = Column(Integer)
        avg_relevance_score  = Column(Float)
        top_relevance_score  = Column(Float)
        faithfulness_score   = Column(Float)
        was_rewritten        = Column(Boolean, default=False)
        rewritten_query      = Column(Text, nullable=True)
        answer               = Column(Text, nullable=True)
        model_used           = Column(String(128), nullable=True)
        user_feedback        = Column(Integer, nullable=True)   # 1=good, 0=bad


class FeedbackLogger:
    """
    Logs query interactions to a durable store for offline analysis
    and strategy adaptation.

    Falls back gracefully to no-op logging when SQLAlchemy is not installed.
    """

    def __init__(self, database_url: str | None = None):
        url = database_url or os.getenv("DATABASE_URL", _DEFAULT_DB)
        self._available = _HAS_SQLALCHEMY

        if not self._available:
            logger.warning(
                "feedback_logger_sqlalchemy_missing",
                extra={"detail": "Install sqlalchemy to enable feedback logging."},
            )
            return

        try:
            engine = create_engine(url, echo=False)
            _Base.metadata.create_all(engine)
            self.session = Session(engine)
            logger.info("feedback_logger_ready", extra={"db": url})
        except Exception as exc:
            logger.error("feedback_logger_init_error", extra={"error": str(exc)})
            self._available = False

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def log(
        self,
        query:                str,
        query_type:           str           = "factual",
        strategy_used:        str           = "dense",
        top_k:                int           = 5,
        num_chunks_retrieved: int           = 0,
        num_chunks_passed:    int           = 0,
        avg_relevance_score:  float         = 0.0,
        top_relevance_score:  float         = 0.0,
        faithfulness_score:   float         = 0.0,
        was_rewritten:        bool          = False,
        rewritten_query:      Optional[str] = None,
        answer:               Optional[str] = None,
        model_used:           Optional[str] = None,
        user_feedback:        Optional[int] = None,
    ) -> Optional[int]:
        """
        Persist one query interaction.

        Returns the integer row ID on success, or None if logging is unavailable.
        """
        if not self._available:
            return None

        try:
            entry = QueryLog(
                query                = query,
                query_type           = query_type,
                strategy_used        = strategy_used,
                top_k                = top_k,
                num_chunks_retrieved = num_chunks_retrieved,
                num_chunks_passed    = num_chunks_passed,
                avg_relevance_score  = avg_relevance_score,
                top_relevance_score  = top_relevance_score,
                faithfulness_score   = faithfulness_score,
                was_rewritten        = was_rewritten,
                rewritten_query      = rewritten_query,
                answer               = answer,
                model_used           = model_used,
                user_feedback        = user_feedback,
            )
            self.session.add(entry)
            self.session.commit()
            self.session.refresh(entry)
            return entry.id
        except Exception as exc:
            logger.error("feedback_logger_log_error", extra={"error": str(exc)})
            self.session.rollback()
            return None

    def add_feedback(self, log_id: int, feedback: int):
        """
        Attach user feedback (1 = good, 0 = bad) to an existing log entry.
        Useful for a /feedback endpoint.
        """
        if not self._available:
            return
        try:
            entry = self.session.get(QueryLog, log_id)
            if entry:
                entry.user_feedback = feedback
                self.session.commit()
        except Exception as exc:
            logger.error("feedback_logger_feedback_error", extra={"error": str(exc)})
            self.session.rollback()

    def get_strategy_stats(self) -> dict:
        """
        Aggregate performance stats grouped by (query_type, strategy_used).
        Used by AdaptationEngine to pick the best strategy per type.

        Returns:
            {
              "factual:dense":     {"count": 42, "avg_relevance": 0.81, ...},
              "reasoning:hybrid":  {...},
              ...
            }
        """
        if not self._available:
            return {}

        try:
            rows = (
                self.session.query(
                    QueryLog.query_type,
                    QueryLog.strategy_used,
                    func.count(QueryLog.id).label("count"),
                    func.avg(QueryLog.avg_relevance_score).label("avg_relevance"),
                    func.avg(QueryLog.faithfulness_score).label("avg_faithfulness"),
                    func.avg(QueryLog.user_feedback).label("avg_feedback"),
                )
                .group_by(QueryLog.query_type, QueryLog.strategy_used)
                .all()
            )
            return {
                f"{row.query_type}:{row.strategy_used}": {
                    "count":            row.count,
                    "avg_relevance":    round(row.avg_relevance    or 0.0, 3),
                    "avg_faithfulness": round(row.avg_faithfulness or 0.0, 3),
                    "avg_feedback":     round(row.avg_feedback     or 0.0, 3),
                }
                for row in rows
            }
        except Exception as exc:
            logger.error("feedback_logger_stats_error", extra={"error": str(exc)})
            return {}

    def close(self):
        if self._available:
            try:
                self.session.close()
            except Exception:
                pass
