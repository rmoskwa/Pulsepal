"""
Session log rotation and management system for PulsePal.

Handles automatic rotation, archival, and cleanup of session logs
to prevent unbounded disk usage while preserving important sessions.
"""

import asyncio
import gzip
import json
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LogRotationManager:
    """Manages rotation, archival, and cleanup of session logs."""
    
    def __init__(
        self,
        log_dir: str = "conversationLogs",
        archive_dir: str = "conversationLogs/archive",
        retention_days: int = 30,
        max_size_gb: float = 1.0,
        importance_threshold: float = 0.7,
    ):
        """
        Initialize the log rotation manager.
        
        Args:
            log_dir: Directory containing session logs
            archive_dir: Directory for archived sessions
            retention_days: Number of days to retain sessions
            max_size_gb: Maximum total size of logs in GB
            importance_threshold: Score threshold for archiving important sessions
        """
        self.log_dir = Path(log_dir)
        self.archive_dir = Path(archive_dir)
        self.retention_days = retention_days
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.importance_threshold = importance_threshold
        
        # Ensure directories exist
        self.log_dir.mkdir(exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        # File locking for thread safety
        self._lock = asyncio.Lock()
        
    def _parse_session_timestamp(self, filename: str) -> Optional[datetime]:
        """
        Extract timestamp from session filename.
        
        Handles both formats:
        - session_20240115_143022.jsonl (timestamp format)
        - session_abc123.jsonl (hex ID format - use file modification time)
        """
        try:
            # Try to extract timestamp from filename
            if "_20" in filename:  # Likely a timestamp format
                parts = filename.split("_")
                date_part = None
                time_part = None
                
                for i, part in enumerate(parts):
                    # Handle date part (with or without extension)
                    clean_part = part.split(".")[0]  # Remove extension if present
                    
                    if clean_part.startswith("20") and len(clean_part) >= 8:
                        date_part = clean_part[:8]
                        
                        # Check if time is in same part (e.g., 20240115143022)
                        if len(clean_part) == 14:
                            time_part = clean_part[8:14]
                        # Check if time is partial in same part (e.g., 20240115_14)
                        elif len(clean_part) > 8:
                            time_suffix = clean_part[8:]
                            if time_suffix.isdigit():
                                # Pad with zeros to make 6 digits
                                time_part = time_suffix.ljust(6, '0')
                        # Check if next part is time
                        elif i + 1 < len(parts):
                            next_part = parts[i + 1].split(".")[0]  # Remove extension
                            if next_part.isdigit():
                                if len(next_part) == 6:
                                    time_part = next_part
                                elif len(next_part) < 6:
                                    # Partial time, pad with zeros
                                    time_part = next_part.ljust(6, '0')
                        break
                
                if date_part:
                    if time_part:
                        timestamp_str = f"{date_part}_{time_part}"
                        return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    else:
                        # Date only, no time
                        return datetime.strptime(date_part, "%Y%m%d")
            
            # Fallback to file modification time
            file_path = self.log_dir / filename
            if file_path.exists():
                mtime = file_path.stat().st_mtime
                return datetime.fromtimestamp(mtime)
                
        except Exception as e:
            logger.debug(f"Could not parse timestamp from {filename}: {e}")
        
        return None
    
    def calculate_directory_size(self) -> int:
        """Calculate total size of log directory in bytes."""
        total_size = 0
        for file_path in self.log_dir.glob("session_*.jsonl"):
            # Skip search files (they have _searches.jsonl suffix)
            if "_searches.jsonl" in file_path.name:
                continue
                
            if file_path.is_file():
                total_size += file_path.stat().st_size
                # Also count associated files
                base_name = file_path.stem
                txt_file = self.log_dir / f"{base_name}.txt"
                search_file = self.log_dir / f"{base_name}_searches.jsonl"
                
                if txt_file.exists():
                    total_size += txt_file.stat().st_size
                if search_file.exists():
                    total_size += search_file.stat().st_size
                    
        return total_size
    
    def identify_old_sessions(self) -> List[str]:
        """
        Find sessions older than retention period.
        
        Returns:
            List of session IDs eligible for cleanup
        """
        old_sessions = []
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        for file_path in self.log_dir.glob("session_*.jsonl"):
            # Skip search files (they have _searches.jsonl suffix)
            if "_searches.jsonl" in file_path.name:
                continue
                
            timestamp = self._parse_session_timestamp(file_path.name)
            if timestamp and timestamp < cutoff_date:
                # Extract session ID from filename
                session_id = file_path.stem  # e.g., "session_20240115_143022"
                old_sessions.append(session_id)
                
        return old_sessions
    
    def identify_sessions_by_size(self, target_size_bytes: int) -> List[Tuple[str, int]]:
        """
        Identify sessions to remove to meet size target.
        
        Args:
            target_size_bytes: Target size to achieve after cleanup
            
        Returns:
            List of (session_id, size) tuples to remove, oldest first
        """
        sessions_with_size = []
        
        for file_path in self.log_dir.glob("session_*.jsonl"):
            # Skip search files (they have _searches.jsonl suffix)
            if "_searches.jsonl" in file_path.name:
                continue
                
            session_id = file_path.stem
            timestamp = self._parse_session_timestamp(file_path.name)
            
            # Calculate total size for this session
            session_size = file_path.stat().st_size
            txt_file = self.log_dir / f"{session_id}.txt"
            search_file = self.log_dir / f"{session_id}_searches.jsonl"
            
            if txt_file.exists():
                session_size += txt_file.stat().st_size
            if search_file.exists():
                session_size += search_file.stat().st_size
                
            sessions_with_size.append((session_id, session_size, timestamp))
        
        # Sort by timestamp (oldest first)
        sessions_with_size.sort(key=lambda x: x[2] or datetime.min)
        
        # Select sessions to remove
        current_size = self.calculate_directory_size()
        sessions_to_remove = []
        
        for session_id, size, _ in sessions_with_size:
            if current_size <= target_size_bytes:
                break
            sessions_to_remove.append((session_id, size))
            current_size -= size
            
        return sessions_to_remove
    
    def score_session_importance(self, session_id: str) -> float:
        """
        Calculate importance score for a session.
        
        Scoring criteria:
        - Long conversations (>20 exchanges): +0.4
        - Contains errors/issues: +0.3
        - Recent activity: +0.2
        - Large file size (knowledge-rich): +0.1
        
        Args:
            session_id: Session identifier
            
        Returns:
            Importance score between 0 and 1
        """
        score = 0.0
        
        try:
            # Check main conversation file
            jsonl_file = self.log_dir / f"{session_id}.jsonl"
            if not jsonl_file.exists():
                return 0.0
            
            # Read conversation data
            exchanges = 0
            has_errors = False
            
            with open(jsonl_file, 'r') as f:
                for line in f:
                    exchanges += 1
                    if 'error' in line.lower() or 'exception' in line.lower():
                        has_errors = True
            
            # Score based on conversation length
            if exchanges > 20:
                score += 0.4
            elif exchanges > 10:
                score += 0.2
            
            # Score based on errors
            if has_errors:
                score += 0.3
            
            # Score based on recency
            timestamp = self._parse_session_timestamp(jsonl_file.name)
            if timestamp:
                days_old = (datetime.now() - timestamp).days
                if days_old < 7:
                    score += 0.2
                elif days_old < 14:
                    score += 0.1
            
            # Score based on file size (knowledge-rich sessions)
            file_size = jsonl_file.stat().st_size
            if file_size > 50000:  # >50KB
                score += 0.1
                
        except Exception as e:
            logger.warning(f"Error scoring session {session_id}: {e}")
            
        return min(score, 1.0)
    
    async def archive_session(self, session_id: str) -> bool:
        """
        Archive a session by compressing and moving to archive directory.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successfully archived
        """
        async with self._lock:
            try:
                # Create archive subdirectory by year-month
                # Check if session_id has timestamp format
                has_timestamp = False
                if "_20" in session_id:
                    parts = session_id.split("_")
                    for part in parts:
                        if part.startswith("20") and len(part) >= 8 and part[:8].isdigit():
                            has_timestamp = True
                            break
                
                if has_timestamp:
                    timestamp = self._parse_session_timestamp(f"{session_id}.jsonl")
                    if timestamp:
                        archive_subdir = self.archive_dir / timestamp.strftime("%Y-%m")
                    else:
                        archive_subdir = self.archive_dir / "unknown"
                else:
                    # Hex ID format sessions go to "unknown"
                    archive_subdir = self.archive_dir / "unknown"
                    
                archive_subdir.mkdir(parents=True, exist_ok=True)
                
                # Files to archive
                files_to_archive = []
                for pattern in [f"{session_id}.jsonl", f"{session_id}.txt", f"{session_id}_searches.jsonl"]:
                    file_path = self.log_dir / pattern
                    if file_path.exists():
                        files_to_archive.append(file_path)
                
                if not files_to_archive:
                    logger.warning(f"No files found for session {session_id}")
                    return False
                
                # Create compressed archive
                archive_path = archive_subdir / f"{session_id}.tar.gz"
                
                # Use asyncio to run compression in executor
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self._compress_files,
                    files_to_archive,
                    archive_path
                )
                
                # Create metadata file
                metadata = {
                    "session_id": session_id,
                    "archived_at": datetime.now().isoformat(),
                    "importance_score": self.score_session_importance(session_id),
                    "files": [f.name for f in files_to_archive],
                    "original_size": sum(f.stat().st_size for f in files_to_archive),
                    "compressed_size": archive_path.stat().st_size,
                }
                
                metadata_path = archive_subdir / f"{session_id}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Archived session {session_id} to {archive_path}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to archive session {session_id}: {e}")
                return False
    
    def _compress_files(self, files: List[Path], archive_path: Path):
        """Compress multiple files into a tar.gz archive."""
        import tarfile
        
        with tarfile.open(archive_path, "w:gz") as tar:
            for file_path in files:
                tar.add(file_path, arcname=file_path.name)
    
    async def cleanup_session(self, session_id: str) -> bool:
        """
        Remove session files from log directory.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successfully removed
        """
        async with self._lock:
            try:
                removed_files = []
                for pattern in [f"{session_id}.jsonl", f"{session_id}.txt", f"{session_id}_searches.jsonl"]:
                    file_path = self.log_dir / pattern
                    if file_path.exists():
                        file_path.unlink()
                        removed_files.append(pattern)
                
                if removed_files:
                    logger.info(f"Removed session {session_id}: {removed_files}")
                    return True
                else:
                    logger.warning(f"No files found for session {session_id}")
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to cleanup session {session_id}: {e}")
                return False
    
    async def cleanup_old_sessions(
        self,
        dry_run: bool = False,
        archive_important: bool = True
    ) -> Dict[str, any]:
        """
        Main cleanup method that removes old sessions.
        
        Args:
            dry_run: If True, only simulate cleanup without actual changes
            archive_important: If True, archive important sessions before deletion
            
        Returns:
            Dictionary with cleanup statistics
        """
        stats = {
            "sessions_checked": 0,
            "sessions_archived": 0,
            "sessions_removed": 0,
            "space_freed_bytes": 0,
            "errors": [],
        }
        
        try:
            # Identify sessions to cleanup
            old_sessions = self.identify_old_sessions()
            stats["sessions_checked"] = len(old_sessions)
            
            logger.info(f"Found {len(old_sessions)} sessions older than {self.retention_days} days")
            
            for session_id in old_sessions:
                try:
                    # Calculate space that will be freed
                    session_size = 0
                    for pattern in [f"{session_id}.jsonl", f"{session_id}.txt", f"{session_id}_searches.jsonl"]:
                        file_path = self.log_dir / pattern
                        if file_path.exists():
                            session_size += file_path.stat().st_size
                    
                    # Check if session is important
                    importance_score = self.score_session_importance(session_id)
                    
                    if not dry_run:
                        # Archive if important
                        if archive_important and importance_score >= self.importance_threshold:
                            if await self.archive_session(session_id):
                                stats["sessions_archived"] += 1
                        
                        # Remove session
                        if await self.cleanup_session(session_id):
                            stats["sessions_removed"] += 1
                            stats["space_freed_bytes"] += session_size
                    else:
                        # Dry run - just log what would happen
                        action = "archive and remove" if importance_score >= self.importance_threshold else "remove"
                        logger.info(f"[DRY RUN] Would {action} {session_id} (score: {importance_score:.2f}, size: {session_size} bytes)")
                        stats["sessions_removed"] += 1
                        stats["space_freed_bytes"] += session_size
                        if importance_score >= self.importance_threshold:
                            stats["sessions_archived"] += 1
                            
                except Exception as e:
                    error_msg = f"Error processing session {session_id}: {e}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
            
            # Check if size limit exceeded
            current_size = self.calculate_directory_size()
            if current_size > self.max_size_bytes:
                logger.warning(f"Size limit exceeded: {current_size / 1024**3:.2f}GB > {self.max_size_bytes / 1024**3:.2f}GB")
                
                # Remove additional sessions by size
                target_size = int(self.max_size_bytes * 0.8)  # Target 80% of limit
                sessions_to_remove = self.identify_sessions_by_size(target_size)
                
                for session_id, size in sessions_to_remove:
                    if session_id not in old_sessions:  # Don't double-count
                        try:
                            importance_score = self.score_session_importance(session_id)
                            
                            if not dry_run:
                                if archive_important and importance_score >= self.importance_threshold:
                                    if await self.archive_session(session_id):
                                        stats["sessions_archived"] += 1
                                
                                if await self.cleanup_session(session_id):
                                    stats["sessions_removed"] += 1
                                    stats["space_freed_bytes"] += size
                            else:
                                action = "archive and remove" if importance_score >= self.importance_threshold else "remove"
                                logger.info(f"[DRY RUN] Would {action} {session_id} for size limit (score: {importance_score:.2f}, size: {size} bytes)")
                                stats["sessions_removed"] += 1
                                stats["space_freed_bytes"] += size
                                if importance_score >= self.importance_threshold:
                                    stats["sessions_archived"] += 1
                                    
                        except Exception as e:
                            error_msg = f"Error processing session {session_id} for size limit: {e}"
                            logger.error(error_msg)
                            stats["errors"].append(error_msg)
            
            # Log summary
            logger.info(f"Cleanup complete: Archived {stats['sessions_archived']}, Removed {stats['sessions_removed']}, Freed {stats['space_freed_bytes'] / 1024**2:.2f}MB")
            
        except Exception as e:
            error_msg = f"Cleanup failed: {e}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
        
        return stats
    
    async def rotate_logs(self):
        """
        Main entry point for log rotation.
        Called periodically to enforce retention and size policies.
        """
        logger.info("Starting log rotation")
        stats = await self.cleanup_old_sessions(dry_run=False, archive_important=True)
        
        if stats["errors"]:
            logger.error(f"Log rotation completed with {len(stats['errors'])} errors")
        else:
            logger.info("Log rotation completed successfully")
        
        return stats
    
    async def restore_from_archive(self, session_id: str, target_dir: Optional[str] = None) -> bool:
        """
        Restore a session from archive.
        
        Args:
            session_id: Session identifier to restore
            target_dir: Directory to restore to (default: original log directory)
            
        Returns:
            True if successfully restored
        """
        try:
            # Find archive file
            archive_file = None
            for archive_path in self.archive_dir.rglob(f"{session_id}.tar.gz"):
                archive_file = archive_path
                break
            
            if not archive_file:
                logger.error(f"Archive not found for session {session_id}")
                return False
            
            # Extract to target directory
            target = Path(target_dir) if target_dir else self.log_dir
            
            # Use asyncio to run extraction in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._extract_archive,
                archive_file,
                target
            )
            
            logger.info(f"Restored session {session_id} from archive")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore session {session_id}: {e}")
            return False
    
    def _extract_archive(self, archive_path: Path, target_dir: Path):
        """Extract a tar.gz archive to target directory."""
        import tarfile
        
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(target_dir)