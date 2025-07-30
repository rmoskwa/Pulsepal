"""
Dependencies and session management for Pulsepal multi-agent system.

Provides session state management and dependency injection containers 
for both Pulsepal and MRI Expert agents with native RAG integration.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging
import asyncio
import os
import time
from .settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """Session memory for conversation history and code examples."""
    
    session_id: str
    conversation_count: int = 0
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    code_examples: List[Dict[str, Any]] = field(default_factory=list)
    preferred_language: Optional[str] = "matlab"  # Default to MATLAB, can be 'matlab', 'octave', or 'python'
    session_start_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    def add_conversation(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add conversation entry with automatic history management."""
        settings = get_settings()
        
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.conversation_history.append(entry)
        self.conversation_count += 1
        self.last_activity = datetime.now()
        
        # Trim history if too long
        if len(self.conversation_history) > settings.max_conversation_history:
            self.conversation_history = self.conversation_history[-settings.max_conversation_history:]
    
    def add_code_example(self, code: str, language: str, sequence_type: str, 
                        description: Optional[str] = None):
        """Add code example with metadata."""
        settings = get_settings()
        
        example = {
            "code": code,
            "language": language.lower(),
            "sequence_type": sequence_type,
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
        
        self.code_examples.append(example)
        
        # Trim examples if too many
        if len(self.code_examples) > settings.max_code_examples:
            self.code_examples = self.code_examples[-settings.max_code_examples:]
    
    def get_recent_conversations(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation entries."""
        return self.conversation_history[-count:] if self.conversation_history else []
    
    def get_code_examples_by_language(self, language: str) -> List[Dict[str, Any]]:
        """Get code examples filtered by language."""
        return [ex for ex in self.code_examples if ex["language"] == language.lower()]
    
    def detect_language_preference(self, content: str) -> Optional[str]:
        """Detect language preference from user content, defaulting to MATLAB."""
        content_lower = content.lower()
        
        # Simple keyword-based detection
        matlab_keywords = ["matlab", ".m file", "octave", "function", "end"]
        python_keywords = ["python", ".py", "import", "def", "__init__"]
        
        matlab_score = sum(1 for kw in matlab_keywords if kw in content_lower)
        python_score = sum(1 for kw in python_keywords if kw in content_lower)
        
        if python_score > matlab_score and python_score > 0:
            # Only switch to Python if there's clear Python preference
            self.preferred_language = "python"
        else:
            # Default to MATLAB for all other cases (including ties or no keywords)
            self.preferred_language = "matlab"
        
        return self.preferred_language


@dataclass
class PulsePalDependencies:
    """Dependencies for Pulsepal main agent."""
    
    conversation_context: Optional[ConversationContext] = None
    session_manager: Optional['SessionManager'] = None
    rag_initialized: bool = False
    
    async def initialize_rag_services(self):
        """Initialize RAG services (embeddings and Supabase)."""
        if self.rag_initialized:
            return  # Already initialized
        
        settings = get_settings()
        
        try:
            # Check if embeddings should be pre-initialized
            if os.getenv("INIT_EMBEDDINGS", "false").lower() == "true":
                logger.info("Pre-initializing embedding service...")
                from .embeddings import get_embedding_service
                embedding_service = get_embedding_service()
                logger.info("Embedding service pre-initialized")
            
            self.rag_initialized = True
            logger.info("RAG services initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG services: {e}")
            # Don't raise exception - allow graceful degradation
            self.rag_initialized = False


@dataclass  
class MRIExpertDependencies:
    """Dependencies for MRI Expert sub-agent."""
    
    conversation_context: Optional[ConversationContext] = None
    parent_usage: Optional[Any] = None  # Usage tracking from parent agent


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
        elif session_id in self.sessions:
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
            new_timeout = datetime.now() + timedelta(hours=self.max_session_duration_hours)
            self.session_timeouts[session_id] = new_timeout
    
    def get_expired_sessions(self) -> List[str]:
        """Get list of expired session IDs."""
        now = datetime.now()
        return [
            session_id for session_id, timeout in self.session_timeouts.items()
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