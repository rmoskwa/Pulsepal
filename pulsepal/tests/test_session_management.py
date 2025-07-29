"""
Test session management and conversation context functionality.

Tests for session lifecycle, conversation history, code example tracking,
and cleanup mechanisms.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from ..dependencies import (
    SessionManager, 
    ConversationContext, 
    PulsePalDependencies,
    get_session_manager
)
from ..main_agent import create_pulsepal_session


class TestSessionManager:
    """Test SessionManager functionality."""
    
    @pytest.fixture
    def session_manager(self):
        """Create a fresh SessionManager for testing."""
        return SessionManager()
    
    def test_create_session(self, session_manager):
        """Test session creation."""
        session_id = "test_session_123"
        context = session_manager.create_session(session_id)
        
        assert isinstance(context, ConversationContext)
        assert context.session_id == session_id
        assert session_id in session_manager.sessions
        assert session_id in session_manager.session_timeouts
    
    def test_get_existing_session(self, session_manager):
        """Test retrieving existing session."""
        session_id = "existing_session"
        original_context = session_manager.create_session(session_id)
        
        # Add some content to verify it's the same session
        original_context.add_conversation("user", "Hello")
        
        # Get the session again
        retrieved_context = session_manager.get_session(session_id)
        
        assert retrieved_context.session_id == session_id
        assert len(retrieved_context.conversation_history) == 1
        assert retrieved_context.conversation_history[0]["content"] == "Hello"
    
    def test_session_timeout_extension(self, session_manager):
        """Test that session timeout gets extended on access."""
        session_id = "timeout_test"
        context = session_manager.create_session(session_id)
        
        original_timeout = session_manager.session_timeouts[session_id]
        
        # Access the session (should extend timeout)
        session_manager.get_session(session_id)
        
        new_timeout = session_manager.session_timeouts[session_id]
        assert new_timeout > original_timeout
    
    def test_session_expiration(self, session_manager):
        """Test session expiration logic."""
        session_id = "expiring_session"
        context = session_manager.create_session(session_id)
        
        # Manually set timeout to past time
        past_time = datetime.now() - timedelta(hours=1)
        session_manager.session_timeouts[session_id] = past_time
        
        assert session_manager.is_session_expired(session_id)
        
        # Getting expired session should create new one
        new_context = session_manager.get_session(session_id)
        assert new_context.conversation_count == 0  # Fresh session
    
    def test_cleanup_session(self, session_manager):
        """Test session cleanup."""
        session_id = "cleanup_test"
        session_manager.create_session(session_id)
        
        assert session_id in session_manager.sessions
        assert session_id in session_manager.session_timeouts
        
        session_manager.cleanup_session(session_id)
        
        assert session_id not in session_manager.sessions
        assert session_id not in session_manager.session_timeouts
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self, session_manager):
        """Test cleanup of multiple expired sessions."""
        # Create multiple sessions
        session_ids = ["expired_1", "expired_2", "valid_session"]
        for sid in session_ids:
            session_manager.create_session(sid)
        
        # Set first two as expired
        past_time = datetime.now() - timedelta(hours=1)
        session_manager.session_timeouts["expired_1"] = past_time
        session_manager.session_timeouts["expired_2"] = past_time
        
        await session_manager.cleanup_expired_sessions()
        
        # Only valid session should remain
        assert "valid_session" in session_manager.sessions
        assert "expired_1" not in session_manager.sessions
        assert "expired_2" not in session_manager.sessions


class TestConversationContext:
    """Test ConversationContext functionality."""
    
    @pytest.fixture
    def context(self):
        """Create a ConversationContext for testing."""
        return ConversationContext(session_id="test_context")
    
    def test_add_conversation(self, context):
        """Test adding conversation entries."""
        context.add_conversation("user", "Hello, Pulsepal!")
        context.add_conversation("assistant", "Hello! How can I help with Pulseq?")
        
        assert context.conversation_count == 2
        assert len(context.conversation_history) == 2
        assert context.conversation_history[0]["role"] == "user"
        assert context.conversation_history[1]["role"] == "assistant"
    
    def test_conversation_history_limit(self, context):
        """Test conversation history trimming."""
        # Mock settings to have small history limit
        with patch('pulsepal.dependencies.get_settings') as mock_settings:
            mock_settings.return_value.max_conversation_history = 5
            
            # Add more conversations than limit
            for i in range(10):
                context.add_conversation("user", f"Message {i}")
            
            # Should only keep last 5
            assert len(context.conversation_history) == 5
            assert context.conversation_history[-1]["content"] == "Message 9"
    
    def test_add_code_example(self, context):
        """Test adding code examples."""
        matlab_code = """
        seq = mr.Sequence();
        seq.addBlock(mr.makeRfPulse(pi/2, 'Duration', 1e-3));
        """
        
        context.add_code_example(
            matlab_code,
            "matlab",
            "spin_echo",
            "Basic spin echo RF pulse"
        )
        
        assert len(context.code_examples) == 1
        assert context.code_examples[0]["language"] == "matlab"
        assert context.code_examples[0]["sequence_type"] == "spin_echo"
        assert "mr.Sequence()" in context.code_examples[0]["code"]
    
    def test_code_examples_limit(self, context):
        """Test code examples trimming."""
        with patch('pulsepal.dependencies.get_settings') as mock_settings:
            mock_settings.return_value.max_code_examples = 3
            
            # Add more examples than limit
            for i in range(5):
                context.add_code_example(
                    f"code_{i}",
                    "matlab", 
                    "test",
                    f"Example {i}"
                )
            
            # Should only keep last 3
            assert len(context.code_examples) == 3
            assert context.code_examples[-1]["code"] == "code_4"
    
    def test_get_code_examples_by_language(self, context):
        """Test filtering code examples by language."""
        # Add examples in different languages
        context.add_code_example("matlab_code", "matlab", "spin_echo")
        context.add_code_example("python_code", "python", "spin_echo")
        context.add_code_example("octave_code", "octave", "gradient_echo")
        
        matlab_examples = context.get_code_examples_by_language("matlab")
        python_examples = context.get_code_examples_by_language("python")
        
        assert len(matlab_examples) == 1
        assert matlab_examples[0]["code"] == "matlab_code"
        
        assert len(python_examples) == 1
        assert python_examples[0]["code"] == "python_code"
    
    def test_language_preference_detection(self, context):
        """Test automatic language preference detection."""
        # Test MATLAB detection
        matlab_content = "I need help with a MATLAB function and .m files"
        detected = context.detect_language_preference(matlab_content)
        assert detected == "matlab"
        assert context.preferred_language == "matlab"
        
        # Test Python detection
        python_content = "Show me Python code with import statements and def functions"
        detected = context.detect_language_preference(python_content)
        assert detected == "python"
        assert context.preferred_language == "python"
        
        # Test no clear preference
        neutral_content = "Help me understand MRI physics"
        detected = context.detect_language_preference(neutral_content)
        # Should maintain previous preference
        assert context.preferred_language == "python"
    
    def test_get_recent_conversations(self, context):
        """Test getting recent conversation entries."""
        # Add several conversations
        for i in range(10):
            context.add_conversation("user", f"Question {i}")
            context.add_conversation("assistant", f"Answer {i}")
        
        # Get recent conversations
        recent = context.get_recent_conversations(3)
        
        assert len(recent) == 3
        assert recent[-1]["content"] == "Answer 9"
        assert recent[0]["content"] == "Answer 7"


class TestSessionIntegration:
    """Test integration between sessions and agent functionality."""
    
    @pytest.mark.asyncio
    async def test_create_pulsepal_session_integration(self):
        """Test full session creation integration."""
        session_id, deps = await create_pulsepal_session("integration_test")
        
        assert session_id == "integration_test"
        assert isinstance(deps, PulsePalDependencies)
        assert isinstance(deps.conversation_context, ConversationContext)
        assert deps.conversation_context.session_id == session_id
    
    @pytest.mark.asyncio
    async def test_session_manager_singleton(self):
        """Test that session manager is properly singleton."""
        manager1 = get_session_manager()
        manager2 = get_session_manager()
        
        assert manager1 is manager2
        
        # Create session in one instance
        session_id = "singleton_test"
        manager1.create_session(session_id)
        
        # Should be accessible from other instance
        assert session_id in manager2.sessions
    
    @pytest.mark.asyncio
    async def test_conversation_continuity_across_calls(self):
        """Test that conversation history persists across multiple agent calls."""
        session_id, deps = await create_pulsepal_session("continuity_test")
        
        # Simulate first interaction
        deps.conversation_context.add_conversation("user", "What is a spin echo?")
        deps.conversation_context.add_conversation("assistant", "A spin echo is...")
        
        # Get session again (simulating new request)
        session_manager = get_session_manager()
        context = session_manager.get_session(session_id)
        
        # Should have conversation history
        assert len(context.conversation_history) == 2
        assert context.conversation_history[0]["content"] == "What is a spin echo?"
    
    def test_session_timeout_configuration(self):
        """Test session timeout configuration from settings."""
        with patch('pulsepal.dependencies.get_settings') as mock_settings:
            mock_settings.return_value.max_session_duration_hours = 12
            
            manager = SessionManager()
            assert manager.max_session_duration_hours == 12
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_on_session_end(self):
        """Test that memory is properly cleaned up when sessions end."""
        session_manager = SessionManager()
        
        # Create and populate a session
        session_id = "memory_test"
        context = session_manager.create_session(session_id)
        
        # Add significant content
        for i in range(100):
            context.add_conversation("user", f"Long message {i}" * 10)
        
        # Cleanup session
        session_manager.cleanup_session(session_id)
        
        # Verify cleanup
        assert session_id not in session_manager.sessions
        assert session_id not in session_manager.session_timeouts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])