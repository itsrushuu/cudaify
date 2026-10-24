"""Rate limiter for CUDAify.

Uses slowapi with in-memory storage. Suitable for single-instance
deployments. Multi-instance deployments should replace storage_uri with
a Redis-backed storage (e.g. "redis://localhost:6379") and share the
limiter instance across processes.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")

__all__ = ["limiter"]
