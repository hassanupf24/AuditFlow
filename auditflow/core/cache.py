# ============================================================
# auditflow/core/cache.py
# Redis Caching & Pipeline State Management
# ============================================================

import json
import logging
from typing import Any, Optional

import redis
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

class RedisSettings(BaseSettings):
    redis_url: str = "redis://localhost:6379/0"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

class RedisCache:
    """
    Interface for Redis to store pipeline execution state and intermediate results.
    """
    def __init__(self) -> None:
        settings = RedisSettings()
        self.redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        try:
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {settings.redis_url}")
        except redis.ConnectionError as e:
            logger.warning(f"Could not connect to Redis: {e}")

    def set_pipeline_status(self, job_id: str, status: str, progress: str = "") -> None:
        """
        Set the current status and progress of a pipeline job.
        status: PENDING, RUNNING, SUCCESS, FAILED
        """
        data = {"status": status, "progress": progress}
        try:
            self.redis_client.set(f"pipeline:{job_id}", json.dumps(data), ex=86400) # Expire in 24h
        except redis.RedisError as e:
            logger.error(f"Redis set error: {e}")

    def get_pipeline_status(self, job_id: str) -> Optional[dict[str, Any]]:
        """
        Get the current status of a pipeline job.
        """
        try:
            data = self.redis_client.get(f"pipeline:{job_id}")
            if data:
                return json.loads(data)
        except redis.RedisError as e:
            logger.error(f"Redis get error: {e}")
        return None
