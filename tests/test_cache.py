import pytest
from auditflow.core.cache import RedisCache

def test_cache_init(monkeypatch):
    """Test that RedisCache initializes correctly and fails gracefully if connection is bad."""
    monkeypatch.setenv("REDIS_URL", "redis://localhost:9999/0")
    cache = RedisCache()
    # Should not crash, just logs a warning
    assert cache.redis_client is not None

def test_set_get_status(monkeypatch, mocker):
    """Test setting and getting pipeline status."""
    cache = RedisCache()
    
    # Mock redis client methods
    mocker.patch.object(cache.redis_client, "set", return_value=True)
    mocker.patch.object(cache.redis_client, "get", return_value='{"status": "RUNNING", "progress": "Cleaning"}')
    
    cache.set_pipeline_status("test-job-123", "RUNNING", "Cleaning")
    cache.redis_client.set.assert_called_once()
    
    status = cache.get_pipeline_status("test-job-123")
    assert status is not None
    assert status["status"] == "RUNNING"
    assert status["progress"] == "Cleaning"
