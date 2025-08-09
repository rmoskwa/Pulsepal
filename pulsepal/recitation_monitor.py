"""
Monitor for RECITATION errors - treats them as critical system failures.

RECITATION errors indicate the agent is attempting to generate code from memory
instead of using database examples. This should NEVER happen with proper prompts.
"""

import logging
from typing import Optional
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


class RecitationMonitor:
    """
    Monitors and alerts on RECITATION errors without hiding them.
    These errors indicate fundamental problems that need immediate attention.
    """

    def __init__(self, alert_threshold: int = 1):
        """
        Initialize monitor.

        Args:
            alert_threshold: Number of recitations before critical alert (default 1)
        """
        self.alert_threshold = alert_threshold
        self.recitation_log_file = "recitation_errors.jsonl"
        self.recitation_count = 0
        self.prevented_count = 0  # Track prevented RECITATION attempts

    def log_recitation_error(
        self,
        query: str,
        session_id: Optional[str] = None,
        context: Optional[dict] = None,
    ):
        """
        Log a RECITATION error as a critical system failure.

        Args:
            query: The query that caused recitation
            session_id: Session where error occurred
            context: Additional context about the error
        """
        self.recitation_count += 1

        error_record = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "session_id": session_id,
            "context": context,
            "count": self.recitation_count,
            "severity": "CRITICAL",
        }

        # Log to file for analysis
        try:
            with open(self.recitation_log_file, "a") as f:
                json.dump(error_record, f)
                f.write("\n")
        except Exception as e:
            logger.error(f"Failed to write recitation log: {e}")

        # Log as CRITICAL - this should trigger alerts
        logger.critical(
            f"RECITATION ERROR #{self.recitation_count}: "
            f"Agent attempted to generate from memory instead of using tools. "
            f"Query: '{query[:100]}...' | Session: {session_id}"
        )

        # If this happens even once, it's a problem
        if self.recitation_count >= self.alert_threshold:
            logger.critical(
                f"🚨 RECITATION THRESHOLD EXCEEDED: {self.recitation_count} errors. "
                "System prompt or tool usage is broken!"
            )

            # Could trigger additional alerts here:
            # - Send to monitoring service
            # - Email developers
            # - Post to Slack
            # - Create GitHub issue

    def get_error_message(self, query: str) -> str:
        """
        Get an honest error message for the user.

        Args:
            query: The original query

        Returns:
            Error message that doesn't hide the problem
        """
        return f"""I encountered a system error while processing your request.

The system attempted to generate code from memory instead of using verified examples,
which violates our quality standards. This indicates a configuration issue that needs
to be addressed.

Your query: "{query}"

Please try:
1. Rephrasing your request
2. Being more specific about what you need
3. Reporting this issue if it persists

Error code: RECITATION_ERROR"""

    def log_prevented_recitation(
        self, query: str, sequence_type: str, session_id: Optional[str] = None
    ):
        """
        Log when RECITATION was successfully prevented.

        Args:
            query: The user query
            sequence_type: Type of sequence that would have triggered RECITATION
            session_id: Session where prevention occurred
        """
        self.prevented_count += 1

        prevention_record = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "sequence_type": sequence_type,
            "session_id": session_id,
            "prevented_count": self.prevented_count,
            "type": "PREVENTED",
        }

        # Log to file for analysis (same file, different type)
        try:
            with open(self.recitation_log_file, "a") as f:
                json.dump(prevention_record, f)
                f.write("\n")
        except Exception as e:
            logger.error(f"Failed to write prevention log: {e}")

        # Log as INFO - this is good news
        logger.info(
            f"RECITATION PREVENTED #{self.prevented_count}: "
            f"Successfully avoided RECITATION for {sequence_type}. "
            f"Query: '{query[:100]}...' | Session: {session_id}"
        )

    def analyze_patterns(self) -> dict:
        """
        Analyze recitation errors and preventions to identify patterns.

        Returns:
            Analysis of recitation patterns
        """
        if not os.path.exists(self.recitation_log_file):
            return {"total_errors": 0, "total_prevented": 0, "patterns": []}

        errors = []
        preventions = []
        with open(self.recitation_log_file, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("type") == "PREVENTED":
                        preventions.append(record)
                    else:
                        errors.append(record)
                except Exception:
                    continue

        # Analyze patterns
        query_keywords = {}
        prevented_sequences = {}

        for error in errors:
            query_lower = error["query"].lower()
            # Track common keywords in failing queries
            for keyword in ["show", "example", "create", "write", "implement"]:
                if keyword in query_lower:
                    query_keywords[keyword] = query_keywords.get(keyword, 0) + 1

        for prevention in preventions:
            seq_type = prevention.get("sequence_type", "unknown")
            prevented_sequences[seq_type] = prevented_sequences.get(seq_type, 0) + 1

        return {
            "total_errors": len(errors),
            "total_prevented": len(preventions),
            "prevention_rate": len(preventions) / (len(errors) + len(preventions))
            if (errors or preventions)
            else 0,
            "first_error": errors[0]["timestamp"] if errors else None,
            "last_error": errors[-1]["timestamp"] if errors else None,
            "common_patterns": query_keywords,
            "prevented_sequences": prevented_sequences,
            "unique_sessions": len(
                set(e.get("session_id") for e in errors + preventions)
            ),
        }


# Global monitor instance
_monitor = None


def get_recitation_monitor() -> RecitationMonitor:
    """Get or create the global recitation monitor."""
    global _monitor
    if _monitor is None:
        _monitor = RecitationMonitor()
    return _monitor
