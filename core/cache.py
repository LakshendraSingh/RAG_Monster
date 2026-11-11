import redis
from config import REDIS_URL
import json

class CacheManager:
    """Manages Redis connection for caching and short-term chat history with in-memory fallback."""
    def __init__(self):
        self._memory_cache = {}
        self._memory_history = {}
        self.use_redis = False
        self.redis_client = None
        try:
            client = redis.from_url(REDIS_URL, decode_responses=True)
            client.ping()
            self.redis_client = client
            self.use_redis = True
        except Exception:
            self.use_redis = False

    def set_cache(self, key: str, value: str, expiration_seconds: int = 3600):
        """Set a simplistic cache value."""
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.setex(key, expiration_seconds, value)
                return
            except Exception:
                pass
        self._memory_cache[key] = value

    def get_cache(self, key: str):
        """Get cache value."""
        if self.use_redis and self.redis_client:
            try:
                return self.redis_client.get(key)
            except Exception:
                pass
        return self._memory_cache.get(key)
        
    def add_chat_message(self, session_id: str, role: str, message: str):
        """Store chat history."""
        key = f"chat_history:{session_id}"
        msg_obj = {"role": role, "content": message}
        if self.use_redis and self.redis_client:
            try:
                msg = json.dumps(msg_obj)
                self.redis_client.rpush(key, msg)
                self.redis_client.expire(key, 86400) # Expire in 24 hours
                return
            except Exception:
                pass
        if session_id not in self._memory_history:
            self._memory_history[session_id] = []
        self._memory_history[session_id].append(msg_obj)
        
    def get_chat_history(self, session_id: str):
        """Retrieve chat history."""
        key = f"chat_history:{session_id}"
        if self.use_redis and self.redis_client:
            try:
                messages = self.redis_client.lrange(key, 0, -1)
                return [json.loads(msg) for msg in messages]
            except Exception:
                pass
        return list(self._memory_history.get(session_id, []))

    def clear_chat_history(self, session_id: str):
        """Clear specific chat history"""
        key = f"chat_history:{session_id}"
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception:
                pass
        self._memory_history.pop(session_id, None)
