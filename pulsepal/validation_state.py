"""
Validation state management for PulsePal.
Provides thread-safe, session-scoped state for validation pipeline.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, Set
import logging

logger = logging.getLogger(__name__)


class ValidationState:
    """
    Session-scoped validation state that avoids race conditions.
    This is stored in PulsePalDependencies and passed through context.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.custom_function_whitelist: Set[str] = set()
        self.skip_function_validation: bool = False
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        # Retry counters for validation phases
        self.tag_retry_count: int = 0
        self.func_retry_count: int = 0

    def add_to_whitelist(self, functions: Set[str]):
        """Add functions to the whitelist."""
        self.custom_function_whitelist.update(functions)
        self.last_accessed = datetime.now()
        logger.debug(
            f"Added {len(functions)} functions to whitelist for session {self.session_id}"
        )

    def get_whitelist(self) -> Set[str]:
        """Get the current whitelist."""
        self.last_accessed = datetime.now()
        return self.custom_function_whitelist.copy()

    def set_skip_validation(self, skip: bool):
        """Set whether to skip function validation."""
        self.skip_function_validation = skip
        self.last_accessed = datetime.now()

    def should_skip_validation(self) -> bool:
        """Check if function validation should be skipped."""
        self.last_accessed = datetime.now()
        return self.skip_function_validation

    def is_expired(self, ttl_hours: int = 24) -> bool:
        """Check if this state has expired."""
        return datetime.now() - self.created_at > timedelta(hours=ttl_hours)

    def reset_retry_counters(self):
        """Reset retry counters after successful validation."""
        self.tag_retry_count = 0
        self.func_retry_count = 0
        logger.debug(f"Reset retry counters for session {self.session_id}")

    def increment_tag_retry(self) -> int:
        """Increment and return tag retry count."""
        self.tag_retry_count += 1
        return self.tag_retry_count

    def increment_func_retry(self) -> int:
        """Increment and return function retry count."""
        self.func_retry_count += 1
        return self.func_retry_count


class ValidationStateManager:
    """
    Manages validation states for all sessions with automatic cleanup.
    This should be a singleton at the application level.
    """

    def __init__(self, ttl_hours: int = 24):
        self.states: Dict[str, ValidationState] = {}
        self.ttl_hours = ttl_hours
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def get_or_create_state(self, session_id: str) -> ValidationState:
        """Get or create validation state for a session."""
        async with self._lock:
            if session_id not in self.states:
                self.states[session_id] = ValidationState(session_id)
                logger.info(f"Created validation state for session {session_id}")

            state = self.states[session_id]
            state.last_accessed = datetime.now()

            # Start cleanup task if not running
            if self._cleanup_task is None or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

            return state

    async def clear_state(self, session_id: str):
        """Clear validation state for a session."""
        async with self._lock:
            if session_id in self.states:
                del self.states[session_id]
                logger.debug(f"Cleared validation state for session {session_id}")

    async def _periodic_cleanup(self):
        """Periodically clean up expired sessions."""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")

    async def _cleanup_expired(self):
        """Remove expired validation states."""
        async with self._lock:
            expired_sessions = [
                sid
                for sid, state in self.states.items()
                if state.is_expired(self.ttl_hours)
            ]

            for sid in expired_sessions:
                del self.states[sid]

            if expired_sessions:
                logger.info(
                    f"Cleaned up {len(expired_sessions)} expired validation states"
                )


# Global singleton instance
_validation_state_manager: Optional[ValidationStateManager] = None


def get_validation_state_manager() -> ValidationStateManager:
    """Get the global validation state manager."""
    global _validation_state_manager
    if _validation_state_manager is None:
        _validation_state_manager = ValidationStateManager()
    return _validation_state_manager
