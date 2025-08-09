"""
Timeout utilities for Pulsepal to ensure responsive performance.
"""

import asyncio
import functools
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def async_timeout(seconds: int = 10):
    """
    Async timeout decorator that ensures operations complete within time limit.

    Args:
        seconds: Maximum seconds before timeout (default 10)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.warning(f"{func.__name__} timed out after {seconds} seconds")
                # Return a timeout error message
                return f"Operation timed out after {seconds} seconds. Please try a more specific query or check your connection."
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                raise

        return wrapper

    return decorator


class TimeoutContext:
    """Context manager for timeout operations with graceful handling."""

    def __init__(self, seconds: int = 10, error_message: Optional[str] = None):
        self.seconds = seconds
        self.error_message = (
            error_message or f"Operation timed out after {seconds} seconds"
        )
        self.task = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        return False

    async def run(self, coro):
        """Run a coroutine with timeout."""
        try:
            self.task = asyncio.create_task(coro)
            return await asyncio.wait_for(self.task, timeout=self.seconds)
        except asyncio.TimeoutError:
            logger.warning(f"Operation timed out after {self.seconds} seconds")
            return self.error_message
