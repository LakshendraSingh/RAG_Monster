"""
core/semantic_cache.py
─────────────────────
Scoped semantic query cache with index versioning, Redis SCAN, and TTL.

Features:
  - Scoped by session/user/ACL (prevents cross-tenant / cross-session leaks)
  - Keyed by index version (invalidates instantly when documents are re-indexed)
  - Uses unique UUID Redis keys and SCAN instead of blocking KEYS
  - In-memory L1 cache with Redis L2 persistence
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    import math

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Cache RAG answers by query embedding similarity, scoped by user/session/ACL
    and invalidated on index version changes.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        redis_client=None,
        ttl_seconds: int = 3600,
        index_version: int = 1,
    ):
        self.threshold = similarity_threshold
        self.redis_client = redis_client
        self.ttl = ttl_seconds
        self.index_version = index_version

        # In-memory L1 store
        self._mem: list[dict[str, Any]] = []

        self.hits = 0
        self.misses = 0

    def invalidate(self):
        """Bump index version and clear cache (called on document index/delete)."""
        self.index_version += 1
        self._mem.clear()
        if self.redis_client:
            try:
                # Use SCAN instead of KEYS to avoid blocking Redis
                for key in self.redis_client.scan_iter(match="sem_cache:*", count=100):
                    self.redis_client.delete(key)
            except Exception as exc:
                logger.warning("semantic_cache_invalidate_redis_error", extra={"error": str(exc)})
        logger.info("semantic_cache_invalidated", extra={"new_index_version": self.index_version})

    def get(self, query_embedding: list[float], scope: str = "default") -> Optional[dict]:
        """
        Look for a cached entry matching scope, index_version, and similarity.
        """
        best_score = 0.0
        best_result: Optional[dict] = None

        entries = self._all_entries(scope)
        for entry in entries:
            # Enforce index version and scope
            if entry.get("index_version", 1) != self.index_version:
                continue
            if entry.get("scope", "default") != scope:
                continue

            score = self._cosine(query_embedding, entry["embedding"])
            if score > best_score:
                best_score = score
                best_result = entry

        if best_score >= self.threshold and best_result is not None:
            self.hits += 1
            logger.info(
                "semantic_cache_hit",
                extra={
                    "similarity": round(best_score, 4),
                    "scope": scope,
                    "index_version": self.index_version,
                },
            )
            return best_result["result"]

        self.misses += 1
        return None

    def set(
        self,
        query: str,
        query_embedding: list[float],
        result: dict,
        scope: str = "default",
    ):
        """Store entry tied to the current scope and index version."""
        entry = {
            "query": query,
            "embedding": query_embedding,
            "result": result,
            "scope": scope,
            "index_version": self.index_version,
        }
        self._mem.append(entry)

        if self.redis_client:
            try:
                unique_id = uuid.uuid4().hex
                key = f"sem_cache:{self.index_version}:{scope}:{unique_id}"
                self.redis_client.setex(key, self.ttl, json.dumps(entry))
            except Exception as exc:
                logger.warning("semantic_cache_redis_write_error", extra={"error": str(exc)})

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": round(self.hits / max(total, 1), 3),
            "cache_size": len(self._mem),
            "index_version": self.index_version,
        }

    def clear(self):
        self._mem.clear()
        self.hits = 0
        self.misses = 0

    def _all_entries(self, scope: str = "default") -> list[dict]:
        """Merge in-memory entries with Redis entries scanned by scope & version."""
        entries = [
            e for e in self._mem
            if e.get("index_version", 1) == self.index_version and e.get("scope", "default") == scope
        ]

        if self.redis_client:
            try:
                pattern = f"sem_cache:{self.index_version}:{scope}:*"
                # Safe non-blocking iteration using SCAN
                for key in self.redis_client.scan_iter(match=pattern, count=100):
                    raw = self.redis_client.get(key)
                    if raw:
                        entry = json.loads(raw)
                        # Deduplicate by query text
                        if not any(e["query"] == entry["query"] for e in entries):
                            entries.append(entry)
            except Exception as exc:
                logger.warning("semantic_cache_redis_scan_error", extra={"error": str(exc)})

        return entries

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if _HAS_NUMPY:
            va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
            denom = np.linalg.norm(va) * np.linalg.norm(vb)
            if denom < 1e-9:
                return 0.0
            return float(np.dot(va, vb) / denom)
        else:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(y * y for y in b))
            denom = norm_a * norm_b
            if denom < 1e-9:
                return 0.0
            return float(dot / denom)
