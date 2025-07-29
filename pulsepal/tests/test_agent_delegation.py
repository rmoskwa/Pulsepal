"""
Test agent delegation patterns with TestModel and FunctionModel validation.

Comprehensive testing of multi-agent communication, delegation flows,
and proper context passing between Pulsepal and MRI Expert agents.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel, FunctionModel

from ..main_agent import pulsepal_agent, create_pulsepal_session, run_pulsepal
from ..mri_expert_agent import mri_expert_agent, consult_mri_expert
from ..dependencies import PulsePalDependencies, MRIExpertDependencies, ConversationContext
from ..providers import get_test_model


class TestAgentDelegation:
    """Test delegation patterns between Pulsepal and MRI Expert agents."""
    
    @pytest.fixture
    async def mock_pulsepal_deps(self):
        """Create mock dependencies for Pulsepal agent."""
        context = ConversationContext(session_id="test_session")
        deps = PulsePalDependencies(conversation_context=context)
        
        # Mock MCP server
        deps.mcp_server = AsyncMock()
        deps.mcp_server.call_tool = AsyncMock()
        
        return deps
    
    @pytest.fixture
    def mock_mri_expert_deps(self):
        """Create mock dependencies for MRI Expert agent."""
        return MRIExpertDependencies()
    
    @pytest.mark.asyncio
    async def test_pulsepal_agent_instantiation(self):
        """Test that Pulsepal agent can be instantiated with TestModel."""
        test_agent = pulsepal_agent.override(model=TestModel())
        
        # Test basic agent structure
        assert test_agent is not None
        assert test_agent.deps_type == PulsePalDependencies
    
    @pytest.mark.asyncio
    async def test_mri_expert_agent_instantiation(self):
        """Test that MRI Expert agent can be instantiated with TestModel."""
        test_agent = mri_expert_agent.override(model=TestModel())
        
        # Test basic agent structure
        assert test_agent is not None
        assert test_agent.deps_type == MRIExpertDependencies
    
    @pytest.mark.asyncio
    async def test_delegation_tool_registration(self, mock_pulsepal_deps):
        """Test that delegation tool is properly registered."""
        test_agent = pulsepal_agent.override(model=TestModel())
        
        # Check that delegation tool exists
        tools = [tool.name for tool in test_agent._function_tools]
        assert "delegate_to_mri_expert" in tools
    
    @pytest.mark.asyncio
    async def test_rag_tools_registration(self, mock_pulsepal_deps):
        """Test that RAG tools are properly registered."""
        test_agent = pulsepal_agent.override(model=TestModel())
        
        # Check that RAG tools exist
        tools = [tool.name for tool in test_agent._function_tools]
        assert "perform_rag_query" in tools
        assert "search_code_examples" in tools  
        assert "get_available_sources" in tools
    
    @pytest.mark.asyncio
    async def test_mri_expert_consultation(self, mock_mri_expert_deps):
        """Test MRI Expert consultation with TestModel."""
        # Create test model that returns a physics explanation
        test_responses = [
            "T1 relaxation is the longitudinal recovery of magnetization..."
        ]
        test_model = TestModel(*test_responses)
        
        test_agent = mri_expert_agent.override(model=test_model)
        
        # Test consultation
        question = "Explain T1 relaxation"
        with patch('pulsepal.mri_expert_agent.mri_expert_agent', test_agent):
            response = await consult_mri_expert(question)
        
        assert response is not None
        assert "T1 relaxation" in response
    
    @pytest.mark.asyncio
    async def test_delegation_with_function_model(self, mock_pulsepal_deps):
        """Test delegation using FunctionModel for controlled behavior."""
        
        # Create function that simulates delegation behavior
        async def mock_delegate_call(question: str, context: str = None) -> str:
            return f"MRI Expert response to: {question}"
        
        function_model = FunctionModel(mock_delegate_call)
        test_agent = pulsepal_agent.override(model=function_model)
        
        # Mock the MCP server to avoid external calls
        mock_pulsepal_deps.mcp_server.call_tool.return_value = {
            "results": [{"title": "Test", "content": "Test content"}]
        }
        
        # Test delegation through the agent
        result = await test_agent.run(
            "What is T1 relaxation?", 
            deps=mock_pulsepal_deps
        )
        
        assert result.data is not None
    
    @pytest.mark.asyncio
    async def test_session_context_delegation(self):
        """Test that session context is properly passed during delegation."""
        session_id, deps = await create_pulsepal_session("test_delegation")
        
        # Add some conversation history
        deps.conversation_context.add_conversation("user", "Tell me about MRI physics")
        deps.conversation_context.add_conversation("assistant", "I'd be happy to help!")
        
        # Create test model for MRI Expert
        test_responses = ["Based on our conversation about MRI physics..."]
        test_model = TestModel(*test_responses)
        
        # Test that context is available
        with patch('pulsepal.mri_expert_agent.mri_expert_agent.override') as mock_override:
            mock_override.return_value.run = AsyncMock(
                return_value=Mock(data="Physics explanation with context")
            )
            
            response = await consult_mri_expert(
                "Explain k-space",
                conversation_history=deps.conversation_context.get_recent_conversations()
            )
            
            assert response is not None
    
    @pytest.mark.asyncio
    async def test_mcp_server_fallback_behavior(self, mock_pulsepal_deps):
        """Test graceful fallback when MCP server is unavailable."""
        # Simulate MCP server failure
        mock_pulsepal_deps.mcp_server = None
        
        test_agent = pulsepal_agent.override(model=TestModel("I'll help with general guidance"))
        
        # Test that agent still responds with fallback
        result = await test_agent.run(
            "Search for spin echo examples",
            deps=mock_pulsepal_deps
        )
        
        assert result.data is not None
    
    @pytest.mark.asyncio
    async def test_language_preference_handling(self):
        """Test that language preferences are detected and maintained."""
        session_id, deps = await create_pulsepal_session("test_language")
        
        # Test MATLAB preference detection
        matlab_query = "Show me a MATLAB function for spin echo"
        deps.conversation_context.detect_language_preference(matlab_query)
        
        assert deps.conversation_context.preferred_language == "matlab"
        
        # Test Python preference detection  
        python_query = "Show me Python code using pulseq package"
        deps.conversation_context.detect_language_preference(python_query)
        
        assert deps.conversation_context.preferred_language == "python"
    
    @pytest.mark.asyncio
    async def test_conversation_history_management(self):
        """Test conversation history tracking and limits."""
        session_id, deps = await create_pulsepal_session("test_history")
        
        # Add multiple conversations
        for i in range(10):
            deps.conversation_context.add_conversation("user", f"Question {i}")
            deps.conversation_context.add_conversation("assistant", f"Answer {i}")
        
        # Check that history is maintained
        history = deps.conversation_context.get_recent_conversations(5)
        assert len(history) == 5
        assert history[-1]["content"] == "Answer 9"
    
    @pytest.mark.asyncio
    async def test_code_example_tracking(self):
        """Test code example storage and retrieval."""
        session_id, deps = await create_pulsepal_session("test_codes")
        
        # Add code examples
        deps.conversation_context.add_code_example(
            "seq = mr.Sequence(); % Create sequence",
            "matlab",
            "spin_echo",
            "Basic spin echo sequence"
        )
        
        deps.conversation_context.add_code_example(
            "seq = pp.Sequence() # Create sequence", 
            "python",
            "spin_echo",
            "Basic spin echo sequence"
        )
        
        # Test language filtering
        matlab_examples = deps.conversation_context.get_code_examples_by_language("matlab")
        assert len(matlab_examples) == 1
        assert "mr.Sequence()" in matlab_examples[0]["code"]
        
        python_examples = deps.conversation_context.get_code_examples_by_language("python")
        assert len(python_examples) == 1
        assert "pp.Sequence()" in python_examples[0]["code"]


class TestMCPIntegration:
    """Test MCP server integration with mocked responses."""
    
    @pytest.mark.asyncio
    async def test_rag_query_tool_with_mock_response(self):
        """Test RAG query tool with mocked MCP response."""
        deps = PulsePalDependencies()
        deps.mcp_server = AsyncMock()
        deps.mcp_server.call_tool.return_value = {
            "results": [
                {
                    "title": "Spin Echo Sequence",
                    "content": "A spin echo sequence uses 90° and 180° pulses...",
                    "source": "pulseq.github.io"
                }
            ]
        }
        
        test_agent = pulsepal_agent.override(model=TestModel("Found relevant documentation"))
        
        # Test tool can be called without errors
        from ..tools import perform_rag_query
        from pydantic_ai import RunContext
        
        ctx = RunContext(deps=deps, usage=Mock())
        result = await perform_rag_query(ctx, "spin echo sequence")
        
        assert "Spin Echo Sequence" in result
        assert "pulseq.github.io" in result
    
    @pytest.mark.asyncio  
    async def test_code_search_tool_with_mock_response(self):
        """Test code search tool with mocked MCP response."""
        deps = PulsePalDependencies()
        deps.mcp_server = AsyncMock()
        deps.mcp_server.call_tool.return_value = {
            "results": [
                {
                    "title": "spin_echo.m",
                    "description": "MATLAB implementation of spin echo",
                    "source": "pulseq-matlab",
                    "language": "matlab"
                }
            ]
        }
        
        from ..tools import search_code_examples
        from pydantic_ai import RunContext
        
        ctx = RunContext(deps=deps, usage=Mock())
        result = await search_code_examples(ctx, "spin echo")
        
        assert "spin_echo.m" in result
        assert "MATLAB" in result
    
    @pytest.mark.asyncio
    async def test_mcp_server_retry_logic(self):
        """Test MCP server retry logic on failures."""
        deps = PulsePalDependencies()
        deps.mcp_server = AsyncMock()
        
        # Mock server to fail twice then succeed
        deps.mcp_server.call_tool.side_effect = [
            Exception("Connection error"),
            Exception("Timeout error"), 
            {"results": [{"title": "Success", "content": "Found it"}]}
        ]
        
        from ..tools import perform_rag_query
        from pydantic_ai import RunContext
        
        ctx = RunContext(deps=deps, usage=Mock())
        result = await perform_rag_query(ctx, "test query")
        
        # Should succeed after retries
        assert "Success" in result
        assert deps.mcp_server.call_tool.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])