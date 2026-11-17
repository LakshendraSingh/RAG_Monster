"""
core/job_store.py
─────────────────
Durable background job status store backed by Redis with in-memory fallback.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class JobStore:
    """Manages background task status in Redis with in-memory fallback."""

    def __init__(self, redis_client=None, ttl_seconds: int = 86400):
        self.redis_client = redis_client
        self.ttl = ttl_seconds
        self._mem: dict[str, dict[str, Any]] = {}

    def set_job(self, job_id: str, data: dict[str, Any]):
        """Persist or update status for a job."""
        self._mem[job_id] = dict(data)
        if self.redis_client:
            try:
                key = f"job:{job_id}"
                self.redis_client.setex(key, self.ttl, json.dumps(data))
            except Exception as exc:
                logger.warning("job_store_redis_write_error", extra={"error": str(exc)})

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Retrieve job status from Redis or memory."""
        if self.redis_client:
            try:
                key = f"job:{job_id}"
                raw = self.redis_client.get(key)
                if raw:
                    return json.loads(raw)
            except Exception as exc:
                logger.warning("job_store_redis_read_error", extra={"error": str(exc)})
        return self._mem.get(job_id)

    def update_job(self, job_id: str, updates: dict[str, Any]):
        """Merge updates into an existing job record."""
        current = self.get_job(job_id) or {}
        current.update(updates)
        self.set_job(job_id, current)
