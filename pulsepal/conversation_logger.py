"""
Conversation logging utilities for debugging PulsePal.

This module provides conversation logging functionality for debugging purposes.
NOTE: This is for debugging only and should be disabled in production.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ConversationLogger:
    """Logger for debugging conversations with PulsePal."""
    
    def __init__(self, log_dir: str = "conversation_logs", enabled: bool = True):
        """
        Initialize conversation logger.
        
        Args:
            log_dir: Directory to store conversation logs
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        
        if self.enabled:
            # Create log directory if it doesn't exist
            self.log_dir.mkdir(exist_ok=True)
            logger.info(f"Conversation logging enabled. Logs will be stored in: {self.log_dir}")
        else:
            logger.info("Conversation logging is disabled")
    
    def log_conversation(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a single conversation turn.
        
        Args:
            session_id: Session identifier
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata (e.g., language preference, tool usage)
        """
        if not self.enabled:
            return
        
        try:
            # Create log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "role": role,
                "content": content,
                "metadata": metadata or {}
            }
            
            # Create session-specific log file
            log_file = self.log_dir / f"session_{session_id[:8]}.jsonl"
            
            # Append to log file (JSONL format for easy parsing)
            with open(log_file, "a", encoding="utf-8") as f:
                json.dump(log_entry, f, ensure_ascii=False)
                f.write("\n")
            
            # Also create a human-readable log
            readable_log = self.log_dir / f"session_{session_id[:8]}.txt"
            with open(readable_log, "a", encoding="utf-8") as f:
                if not readable_log.exists() or readable_log.stat().st_size == 0:
                    f.write(f"=== PulsePal Conversation Log ===\n")
                    f.write(f"Session ID: {session_id}\n")
                    f.write(f"Started: {datetime.now().isoformat()}\n")
                    f.write("=" * 50 + "\n\n")
                
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {role.upper()}:\n")
                f.write(f"{content}\n")
                f.write("-" * 50 + "\n\n")
            
        except Exception as e:
            logger.error(f"Failed to log conversation: {e}")
    
    def log_search_event(
        self,
        session_id: str,
        search_type: str,
        query: str,
        results_count: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log RAG search events for debugging.
        
        Args:
            session_id: Session identifier
            search_type: Type of search (documentation, code, etc.)
            query: Search query
            results_count: Number of results returned
            metadata: Optional metadata
        """
        if not self.enabled:
            return
        
        try:
            # Create search event log
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "event_type": "search",
                "search_type": search_type,
                "query": query,
                "results_count": results_count,
                "metadata": metadata or {}
            }
            
            # Log to session file
            log_file = self.log_dir / f"session_{session_id[:8]}_searches.jsonl"
            with open(log_file, "a", encoding="utf-8") as f:
                json.dump(log_entry, f, ensure_ascii=False)
                f.write("\n")
            
            # Add to readable log
            readable_log = self.log_dir / f"session_{session_id[:8]}.txt"
            with open(readable_log, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SEARCH EVENT:\n")
                f.write(f"  Type: {search_type}\n")
                f.write(f"  Query: {query}\n")
                f.write(f"  Results: {results_count}\n")
                f.write("-" * 50 + "\n\n")
                
        except Exception as e:
            logger.error(f"Failed to log search event: {e}")
    
    def get_session_log(self, session_id: str) -> Optional[str]:
        """
        Read the conversation log for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Log content as string or None if not found
        """
        try:
            readable_log = self.log_dir / f"session_{session_id[:8]}.txt"
            if readable_log.exists():
                return readable_log.read_text(encoding="utf-8")
            return None
        except Exception as e:
            logger.error(f"Failed to read session log: {e}")
            return None
    
    def list_sessions(self) -> list[str]:
        """
        List all logged sessions.
        
        Returns:
            List of session IDs with logs
        """
        try:
            sessions = set()
            for log_file in self.log_dir.glob("session_*.txt"):
                # Extract session ID from filename
                session_id = log_file.stem.replace("session_", "")
                sessions.add(session_id)
            return sorted(list(sessions))
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
    
    def cleanup_old_logs(self, days: int = 7):
        """
        Remove logs older than specified days.
        
        Args:
            days: Number of days to keep logs
        """
        if not self.enabled:
            return
        
        try:
            from datetime import timedelta
            
            cutoff_time = datetime.now() - timedelta(days=days)
            
            for log_file in self.log_dir.glob("session_*"):
                if log_file.stat().st_mtime < cutoff_time.timestamp():
                    log_file.unlink()
                    logger.info(f"Deleted old log file: {log_file}")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup old logs: {e}")


# Global instance
_conversation_logger: Optional[ConversationLogger] = None


def get_conversation_logger() -> ConversationLogger:
    """
    Get the global conversation logger instance.
    
    Returns:
        ConversationLogger instance
    """
    global _conversation_logger
    if _conversation_logger is None:
        # Check environment variable to enable/disable logging
        enabled = os.getenv("ENABLE_CONVERSATION_LOGGING", "false").lower() == "true"
        log_dir = os.getenv("CONVERSATION_LOG_DIR", "conversation_logs")
        _conversation_logger = ConversationLogger(log_dir=log_dir, enabled=enabled)
    return _conversation_logger
