from __future__ import annotations

from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """Simple interface for in-memory result storage."""

    @abstractmethod
    def save(self, key: str, data: str) -> str:
        """Persist ``data`` under ``key`` and return the key."""


class RedisStorage(StorageBackend):
    """Store results in Redis."""

    def __init__(self, client):
        self.client = client

    def save(self, key: str, data: str) -> str:
        self.client.set(key, data)
        return key
