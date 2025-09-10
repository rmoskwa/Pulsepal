"""
Dependencies and session management for Pulsepal.

Provides session state management and dependency injection containers
for both Pulsepal and MRI Expert agents with native RAG integration.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .settings import get_settings

logger = logging.getLogger(__name__)

# Supported programming languages based on database analysis
SUPPORTED_LANGUAGES = {
    "c": {
        "extensions": [".c", ".h"],
        "keywords": ["#include", "int main", "void", "struct"],
        "highlight": "c",
        "description": "C programming language",
    },
    "cpp": {
        "extensions": [".cpp", ".cc", ".cxx", ".hpp"],
        "keywords": ["#include", "class", "namespace", "::", "template"],
        "highlight": "cpp",
        "description": "C++ programming language",
    },
    "julia": {
        "extensions": [".jl"],
        "keywords": ["function", "end", "module", "using"],
        "highlight": "julia",
        "description": "Julia programming language",
    },
    "matlab": {
        "extensions": [".m"],
        "keywords": ["function", "end", "%", "clear", "clc"],
        "highlight": "matlab",
        "description": "MATLAB programming language",
    },
    "octave": {
        "extensions": [".m"],
        "keywords": ["function", "endfunction", "%", "octave"],
        "highlight": "matlab",
        "description": "GNU Octave (MATLAB-compatible)",
    },
    "python": {
        "extensions": [".py"],
        "keywords": ["import", "def", "class", "from", "__init__"],
        "highlight": "python",
        "description": "Python programming language",
    },
}


@dataclass
class ConversationContext:
    """Session memory for conversation history and code examples."""

    session_id: str
    conversation_count: int = 0
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    code_examples: List[Dict[str, Any]] = field(default_factory=list)
    preferred_language: Optional[str] = (
        "matlab"  # Default to MATLAB, can be 'matlab', 'octave', or 'python'
    )
    session_start_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    # Sequence Knowledge fields
    sequence_knowledge: Optional[str] = None
    use_sequence_context: bool = False

    def add_conversation(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ):
        """Add conversation entry with automatic history management."""
        settings = get_settings()

        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        self.conversation_history.append(entry)
        self.conversation_count += 1
        self.last_activity = datetime.now()

        # Trim history if too long
        if len(self.conversation_history) > settings.max_conversation_history:
            self.conversation_history = self.conversation_history[
                -settings.max_conversation_history :
            ]

    def add_code_example(
        self,
        code: str,
        language: str,
        sequence_type: str,
        description: Optional[str] = None,
    ):
        """Add code example with metadata."""
        settings = get_settings()

        example = {
            "code": code,
            "language": language.lower(),
            "sequence_type": sequence_type,
            "description": description,
            "timestamp": datetime.now().isoformat(),
        }

        self.code_examples.append(example)

        # Trim examples if too many
        if len(self.code_examples) > settings.max_code_examples:
            self.code_examples = self.code_examples[-settings.max_code_examples :]

    def get_recent_conversations(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation entries."""
        return self.conversation_history[-count:] if self.conversation_history else []

    def get_formatted_history(self, max_exchanges: int = 5) -> str:
        """
        Format recent conversation history for agent context.

        Args:
            max_exchanges: Maximum number of user-assistant exchanges to include

        Returns:
            Formatted string of recent conversation history
        """
        if not self.conversation_history:
            return ""

        # Filter to only user and assistant messages (exclude system)
        relevant_history = [
            entry
            for entry in self.conversation_history
            if entry["role"] in ["user", "assistant"]
        ]

        if not relevant_history:
            return ""

        # Get last N exchanges (2 messages per exchange)
        recent_messages = relevant_history[-(max_exchanges * 2) :]

        # Format as conversation
        formatted = ["Previous conversation:"]
        for entry in recent_messages:
            role = entry["role"].capitalize()
            content = entry["content"]
            # Truncate very long messages to avoid token limits
            # Increased limit to preserve code context
            if len(content) > 2000:
                content = content[:2000] + "... [truncated]"
            formatted.append(f"{role}: {content}")

        return "\n".join(formatted)

    def get_code_examples_by_language(self, language: str) -> List[Dict[str, Any]]:
        """Get code examples filtered by language."""
        return [ex for ex in self.code_examples if ex["language"] == language.lower()]

    def detect_language_preference(self, content: str) -> Optional[str]:
        """
        Detect programming language preference from query - MATLAB by default.

        Only switches to Python if explicitly indicated.

        Args:
            content: User query or content to analyze

        Returns:
            Language preference (defaults to 'matlab')
        """
        content_lower = content.lower()

        # Strong Python indicators only - must be explicit
        python_indicators = [
            "python",
            "pypulseq",
            ".py",
            "import ",
            "pip ",
            "def ",
            "numpy",
            "__init__",
            "self.",
            "from ",
            "pandas",
            "matplotlib",
            "pyplot",
        ]

        # Only switch to Python if explicitly indicated
        if any(indicator in content_lower for indicator in python_indicators):
            self.preferred_language = "python"
            logger.debug("Detected explicit Python preference in query")
            return "python"

        # Check for other language indicators (C++, Julia, etc.)
        # But still very conservative - require explicit mentions
        if "c++" in content_lower or "cpp" in content_lower:
            self.preferred_language = "cpp"
            logger.debug("Detected explicit C++ preference in query")
            return "cpp"

        if "julia" in content_lower:
            self.preferred_language = "julia"
            logger.debug("Detected explicit Julia preference in query")
            return "julia"

        # DEFAULT TO MATLAB for everything else
        # This includes when users mention:
        # - MATLAB explicitly
        # - Octave
        # - .m files
        # - Or when no language is specified
        self.preferred_language = "matlab"
        logger.debug(
            "Defaulting to MATLAB (no explicit Python/other language indicators)",
        )
        return "matlab"

    def get_supported_languages(self) -> Dict[str, Dict[str, Any]]:
        """Get all supported programming languages with their configurations."""
        return SUPPORTED_LANGUAGES

    def get_language_extensions(self) -> List[str]:
        """Get all supported file extensions."""
        extensions = []
        for lang_config in SUPPORTED_LANGUAGES.values():
            extensions.extend(lang_config["extensions"])
        return list(set(extensions))  # Remove duplicates

    def get_active_context(self) -> Optional[str]:
        """
        Returns formatted sequence context if enabled and available.

        Returns:
            Formatted context string or None if disabled/empty
        """
        if not self.use_sequence_context or not self.sequence_knowledge:
            return None

        return f"""SEQUENCE CONTEXT:
{self.sequence_knowledge}
---
Please consider this sequence-specific context when providing assistance."""

    def export_sequence_knowledge(self) -> str:
        """
        Export sequence knowledge as markdown with metadata.

        Returns:
            Markdown formatted string with sequence knowledge and metadata
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        export_content = f"""# Sequence Knowledge Export
Generated: {timestamp}
Session ID: {self.session_id}
Preferred Language: {self.preferred_language}

## Sequence Context
{self.sequence_knowledge if self.sequence_knowledge else "No sequence context defined."}

## Session Metadata
- Session Start: {self.session_start_time.strftime("%Y-%m-%d %H:%M:%S")}
- Total Conversations: {self.conversation_count}
- Context Enabled: {self.use_sequence_context}
- Last Activity: {self.last_activity.strftime("%Y-%m-%d %H:%M:%S")}
"""

        return export_content


@dataclass
class PulsePalDependencies:
    """Dependencies for Pulsepal main agent."""

    conversation_context: Optional[ConversationContext] = None
    session_manager: Optional["SessionManager"] = None
    rag_initialized: bool = False

    # Validation errors from semantic router (kept for logging)
    validation_errors: Optional[List[str]] = (
        None  # Namespace/function validation errors
    )

    async def initialize_rag_services(self):
        """Initialize RAG services (embeddings and Supabase)."""
        if self.rag_initialized:
            return  # Already initialized

        try:
            # Check if embeddings should be pre-initialized
            if os.getenv("INIT_EMBEDDINGS", "false").lower() == "true":
                logger.info("Pre-initializing embedding service...")
                from .embeddings import get_embedding_service

                get_embedding_service()  # Pre-initialize without storing
                logger.info("Embedding service pre-initialized")

            self.rag_initialized = True
            logger.info("RAG services initialization completed")

        except Exception as e:
            logger.error(f"Failed to initialize RAG services: {e}")
            # Don't raise exception - allow graceful degradation
            self.rag_initialized = False


class SessionManager:
    """Manages session lifecycle and cleanup."""

    def __init__(self):
        self.sessions: Dict[str, ConversationContext] = {}
        self.session_timeouts: Dict[str, datetime] = {}
        self.settings = get_settings()
        self.max_session_duration_hours = self.settings.max_session_duration_hours

    def create_session(self, session_id: str) -> ConversationContext:
        """Create new session with conversation context."""
        context = ConversationContext(session_id=session_id)
        self.sessions[session_id] = context

        # Set session timeout
        timeout = datetime.now() + timedelta(hours=self.max_session_duration_hours)
        self.session_timeouts[session_id] = timeout

        logger.info(f"Created new session: {session_id}")
        return context

    def get_session(self, session_id: str) -> Optional[ConversationContext]:
        """Get existing session or create new one."""
        if session_id in self.sessions and not self.is_session_expired(session_id):
            self.extend_session(session_id)
            return self.sessions[session_id]
        if session_id in self.sessions:
            # Session expired, clean it up
            self.cleanup_session(session_id)

        return self.create_session(session_id)

    def cleanup_session(self, session_id: str):
        """Clean up session data."""
        self.sessions.pop(session_id, None)
        self.session_timeouts.pop(session_id, None)
        logger.info(f"Cleaned up session: {session_id}")

    def is_session_expired(self, session_id: str) -> bool:
        """Check if session has expired."""
        if session_id not in self.session_timeouts:
            return False
        return datetime.now() > self.session_timeouts[session_id]

    def extend_session(self, session_id: str):
        """Extend session timeout on activity."""
        if session_id in self.session_timeouts:
            new_timeout = datetime.now() + timedelta(
                hours=self.max_session_duration_hours,
            )
            self.session_timeouts[session_id] = new_timeout

    def get_expired_sessions(self) -> List[str]:
        """Get list of expired session IDs."""
        now = datetime.now()
        return [
            session_id
            for session_id, timeout in self.session_timeouts.items()
            if now > timeout
        ]

    async def cleanup_expired_sessions(self):
        """Clean up all expired sessions."""
        expired = self.get_expired_sessions()
        for session_id in expired:
            self.cleanup_session(session_id)

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
