import redis
from config import REDIS_URL
import json

class CacheManager:
    """Manages Redis connection for caching and short-term chat history."""
    def __init__(self):
        self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)

    def set_cache(self, key: str, value: str, expiration_seconds: int = 3600):
        """Set a simplistic cache value."""
        self.redis_client.setex(key, expiration_seconds, value)

    def get_cache(self, key: str):
        """Get cache value."""
        return self.redis_client.get(key)
        
    def add_chat_message(self, session_id: str, role: str, message: str):
        """Store chat history."""
        key = f"chat_history:{session_id}"
        msg = json.dumps({"role": role, "content": message})
        self.redis_client.rpush(key, msg)
        self.redis_client.expire(key, 86400) # Expire in 24 hours
        
    def get_chat_history(self, session_id: str):
        """Retrieve chat history."""
        key = f"chat_history:{session_id}"
        messages = self.redis_client.lrange(key, 0, -1)
        return [json.loads(msg) for msg in messages]

    def clear_chat_history(self, session_id: str):
        """Clear specific chat history"""
        key = f"chat_history:{session_id}"
        self.redis_client.delete(key)
