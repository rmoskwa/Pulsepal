"""
Background rotation service for PulsePal session logs.

Integrates with the existing PulsePal application to provide
automatic log rotation in the background.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from .log_manager import LogRotationManager
from .settings import get_settings

logger = logging.getLogger(__name__)


class BackgroundRotationService:
    """Background service for automatic log rotation."""

    def __init__(self, rotation_manager: Optional[LogRotationManager] = None):
        """
        Initialize the background rotation service.

        Args:
            rotation_manager: Optional LogRotationManager instance
        """
        self.settings = get_settings()

        if rotation_manager:
            self.rotation_manager = rotation_manager
        else:
            self.rotation_manager = LogRotationManager(
                retention_days=self.settings.log_retention_days,
                max_size_gb=self.settings.max_log_size_gb,
                importance_threshold=self.settings.importance_threshold,
            )

        self.rotation_interval = self.settings.rotation_check_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def start(self):
        """Start the background rotation service."""
        if self._running:
            logger.warning("Background rotation service is already running")
            return

        self._running = True
        self._shutdown_event.clear()
        self._task = asyncio.create_task(self._rotation_loop())
        logger.info("Background rotation service started")

    async def stop(self):
        """Stop the background rotation service gracefully."""
        if not self._running:
            return

        logger.info("Stopping background rotation service...")
        self._running = False
        self._shutdown_event.set()

        if self._task:
            try:
                # Wait for the task to complete with timeout
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "Background rotation task did not stop in time, cancelling"
                )
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

        logger.info("Background rotation service stopped")

    async def _rotation_loop(self):
        """Main rotation loop that runs in the background."""
        logger.info(f"Starting rotation loop with {self.rotation_interval}s interval")

        while self._running:
            try:
                # Perform rotation
                await self._perform_rotation()

                # Wait for the next interval or shutdown signal
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=self.rotation_interval
                    )
                except asyncio.TimeoutError:
                    # Normal timeout, continue loop
                    pass

            except Exception as e:
                logger.error(f"Error in rotation loop: {e}")
                # Continue running even if rotation fails
                await asyncio.sleep(60)  # Wait a bit before retrying

    async def _perform_rotation(self):
        """Perform a single rotation check and cleanup."""
        try:
            logger.debug("Performing rotation check")

            # Check directory size
            current_size = self.rotation_manager.calculate_directory_size()
            max_size = self.rotation_manager.max_size_bytes

            # Log warning if approaching limits
            usage_percent = (current_size / max_size) * 100 if max_size > 0 else 0

            if usage_percent >= 90:
                logger.warning(
                    f"Log directory at {usage_percent:.1f}% of capacity ({current_size / 1024**2:.1f}MB / {max_size / 1024**2:.1f}MB)"
                )
            elif usage_percent >= 80:
                logger.info(f"Log directory at {usage_percent:.1f}% of capacity")

            # Perform cleanup if needed
            old_sessions = self.rotation_manager.identify_old_sessions()

            if old_sessions or current_size > max_size:
                logger.info(
                    f"Starting automatic cleanup: {len(old_sessions)} old sessions, size: {current_size / 1024**2:.1f}MB"
                )

                stats = await self.rotation_manager.rotate_logs()

                if stats.get("errors"):
                    logger.error(
                        f"Rotation completed with {len(stats['errors'])} errors"
                    )
                else:
                    logger.info(
                        f"Rotation completed: archived {stats['sessions_archived']}, "
                        f"removed {stats['sessions_removed']}, "
                        f"freed {stats['space_freed_bytes'] / 1024**2:.1f}MB"
                    )
            else:
                logger.debug("No rotation needed")

        except Exception as e:
            logger.error(f"Error during rotation: {e}")
            raise

    async def force_rotation(self):
        """Force an immediate rotation check."""
        logger.info("Forcing immediate rotation")
        await self._perform_rotation()

    def get_status(self) -> dict:
        """Get the current status of the rotation service."""
        return {
            "running": self._running,
            "rotation_interval": self.rotation_interval,
            "last_check": datetime.now().isoformat(),
            "directory_size": self.rotation_manager.calculate_directory_size(),
            "max_size": self.rotation_manager.max_size_bytes,
            "retention_days": self.rotation_manager.retention_days,
        }


# Global instance for easy integration
_rotation_service: Optional[BackgroundRotationService] = None


async def start_background_rotation():
    """Start the global background rotation service."""
    global _rotation_service

    if _rotation_service is None:
        _rotation_service = BackgroundRotationService()

    await _rotation_service.start()
    return _rotation_service


async def stop_background_rotation():
    """Stop the global background rotation service."""
    global _rotation_service

    if _rotation_service:
        await _rotation_service.stop()
        _rotation_service = None


def get_rotation_service() -> Optional[BackgroundRotationService]:
    """Get the global rotation service instance."""
    return _rotation_service
