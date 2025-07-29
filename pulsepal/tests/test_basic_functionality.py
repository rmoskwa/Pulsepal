"""
Test basic functionality of Pulsepal agents and core components.

Tests for agent instantiation, tool registration, configuration loading,
and basic response patterns using TestModel validation.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pydantic_ai.models.test import TestModel

from ..main_agent import pulsepal_agent, ask_pulsepal
from ..mri_expert_agent import mri_expert_agent, explain_mri_concept
from ..settings import Settings, load_settings, get_settings
from ..providers import get_llm_model, get_test_model
from ..dependencies import PulsePalDependencies, ConversationContext


class TestConfiguration:
    """Test configuration and settings management."""
    
    def test_settings_model_structure(self):
        """Test that Settings model has required fields."""
        # Mock environment variables
        env_vars = {
            "GOOGLE_API_KEY": "test_key",
            "SUPABASE_URL": "https://test.supabase.co", 
            "SUPABASE_KEY": "test_supabase_key"
        }
        
        with patch.dict('os.environ', env_vars):
            settings = Settings()
            
            assert settings.google_api_key == "test_key"
            assert settings.supabase_url == "https://test.supabase.co"
            assert settings.supabase_key == "test_supabase_key"
            assert settings.llm_model == "gemini-2.0-flash-exp"  # default
    
    def test_settings_validation_errors(self):
        """Test settings validation with missing required fields."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError):
                Settings()
    
    def test_load_settings_with_env_file(self):
        """Test loading settings with .env file support."""
        env_vars = {
            "GOOGLE_API_KEY": "env_file_key",
            "SUPABASE_URL": "https://env.supabase.co",
            "SUPABASE_KEY": "env_supabase_key"
        }
        
        with patch.dict('os.environ', env_vars):
            with patch('pulsepal.settings.load_dotenv'):
                settings = load_settings()
                assert isinstance(settings, Settings)
                assert settings.google_api_key == "env_file_key"
    
    def test_get_settings_caching(self):
        """Test that get_settings() properly caches settings."""
        env_vars = {
            "GOOGLE_API_KEY": "cached_key",
            "SUPABASE_URL": "https://cached.supabase.co",
            "SUPABASE_KEY": "cached_supabase_key"
        }
        
        with patch.dict('os.environ', env_vars):
            # Clear any existing cached settings
            import pulsepal.settings
            pulsepal.settings._settings = None
            
            settings1 = get_settings()
            settings2 = get_settings()
            
            assert settings1 is settings2  # Same instance (cached)


class TestProviders:
    """Test model provider functionality."""
    
    def test_get_test_model(self):
        """Test test model provider."""
        model_name = get_test_model()
        assert model_name == "test"
    
    @patch('pulsepal.providers.get_settings')
    def test_get_llm_model_configuration(self, mock_get_settings):
        """Test LLM model configuration."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.llm_model = "gemini-2.0-flash-exp"
        mock_settings.google_api_key = "test_api_key"
        mock_get_settings.return_value = mock_settings
        
        # This would normally create a GeminiModel, but we'll just test the flow
        with patch('pulsepal.providers.GeminiModel') as mock_gemini:
            get_llm_model()
            mock_gemini.assert_called_once_with(
                model_name="gemini-2.0-flash-exp",
                api_key="test_api_key"
            )


class TestAgentBasics:
    """Test basic agent functionality with TestModel."""
    
    @pytest.mark.asyncio
    async def test_pulsepal_agent_basic_response(self):
        """Test Pulsepal agent can generate basic responses."""
        test_responses = [
            "I can help you with Pulseq MRI sequence programming. What would you like to know?"
        ]
        test_model = TestModel(*test_responses)
        test_agent = pulsepal_agent.override(model=test_model)
        
        # Create minimal dependencies
        deps = PulsePalDependencies(
            conversation_context=ConversationContext(session_id="test")
        )
        
        result = await test_agent.run("Hello", deps=deps)
        assert result.data in test_responses
    
    @pytest.mark.asyncio
    async def test_mri_expert_agent_basic_response(self):
        """Test MRI Expert agent can generate physics explanations."""
        test_responses = [
            "T1 relaxation refers to the longitudinal recovery of magnetization after RF excitation..."
        ]
        test_model = TestModel(*test_responses)
        test_agent = mri_expert_agent.override(model=test_model)
        
        from ..dependencies import MRIExpertDependencies
        deps = MRIExpertDependencies()
        
        result = await test_agent.run("Explain T1 relaxation", deps=deps)
        assert "T1 relaxation" in result.data
    
    def test_pulsepal_agent_tools_registration(self):
        """Test that Pulsepal agent has all required tools registered."""
        expected_tools = [
            "perform_rag_query",
            "search_code_examples", 
            "get_available_sources",
            "delegate_to_mri_expert"
        ]
        
        tool_names = [tool.name for tool in pulsepal_agent._function_tools]
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_names, f"Missing tool: {expected_tool}"
    
    def test_mri_expert_agent_no_tools(self):
        """Test that MRI Expert agent has no tools (pure explanation agent)."""
        tool_names = [tool.name for tool in mri_expert_agent._function_tools]
        assert len(tool_names) == 0, "MRI Expert should not have tools"
    
    @pytest.mark.asyncio
    async def test_ask_pulsepal_simple_interface(self):
        """Test simple ask_pulsepal interface."""
        test_response = "Here's help with your Pulseq question."
        
        with patch('pulsepal.main_agent.run_pulsepal') as mock_run:
            mock_run.return_value = ("session_123", test_response)
            
            response = await ask_pulsepal("Help me with spin echo")
            assert response == test_response
    
    @pytest.mark.asyncio
    async def test_explain_mri_concept_levels(self):
        """Test MRI concept explanation with different detail levels."""
        with patch('pulsepal.mri_expert_agent.consult_mri_expert') as mock_consult:
            mock_consult.return_value = "Basic explanation of k-space..."
            
            # Test basic level
            response = await explain_mri_concept("k-space", "basic")
            mock_consult.assert_called_once()
            call_args = mock_consult.call_args[0][0]
            assert "simple terms" in call_args
            assert "k-space" in call_args
    
    @pytest.mark.asyncio
    async def test_dependencies_initialization(self):
        """Test that dependencies can be properly initialized."""
        deps = PulsePalDependencies()
        context = ConversationContext(session_id="init_test")
        deps.conversation_context = context
        
        # Mock MCP server initialization
        with patch('pulsepal.dependencies.MCPServerStdio') as mock_mcp:
            mock_server = AsyncMock()
            mock_mcp.return_value = mock_server
            
            await deps.initialize_mcp_server()
            
            assert deps.mcp_server is not None
            mock_server.connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mcp_connection_health_check(self):
        """Test MCP connection health checking."""
        deps = PulsePalDependencies()
        
        # Mock successful MCP server
        mock_server = AsyncMock()
        mock_health_checker = AsyncMock()
        mock_health_checker.health_check.return_value = True
        
        deps.mcp_server = mock_server
        deps.health_checker = mock_health_checker
        
        result = await deps.ensure_mcp_connection()
        assert result is True
        mock_health_checker.health_check.assert_called_once()
    
    def test_fallback_responses(self):
        """Test fallback responses for MCP server failures."""
        deps = PulsePalDependencies()
        
        rag_fallback = deps.get_fallback_response("perform_rag_query")
        assert "unable to access the RAG database" in rag_fallback
        
        code_fallback = deps.get_fallback_response("search_code_examples")
        assert "unable to search code examples" in code_fallback
        
        sources_fallback = deps.get_fallback_response("get_available_sources")
        assert "unable to retrieve available sources" in sources_fallback


class TestSystemPrompts:
    """Test system prompts and agent behavior patterns."""
    
    def test_pulsepal_system_prompt_content(self):
        """Test that Pulsepal system prompt contains required elements."""
        from ..main_agent import PULSEPAL_SYSTEM_PROMPT
        
        required_elements = [
            "Pulsepal",
            "Pulseq",
            "MATLAB", 
            "Python",
            "Octave",
            "MRI Expert",
            "perform_rag_query",
            "delegate_to_mri_expert"
        ]
        
        for element in required_elements:
            assert element in PULSEPAL_SYSTEM_PROMPT, f"Missing: {element}"
    
    def test_mri_expert_system_prompt_content(self):
        """Test that MRI Expert system prompt focuses on physics."""
        from ..mri_expert_agent import MRI_EXPERT_SYSTEM_PROMPT
        
        physics_elements = [
            "MRI Expert",
            "physics",
            "k-space",
            "T1", "T2",
            "RF pulse",
            "gradient",
            "educational"
        ]
        
        for element in physics_elements:
            assert element in MRI_EXPERT_SYSTEM_PROMPT, f"Missing physics element: {element}"
    
    def test_agent_role_separation(self):
        """Test that agent roles are properly separated."""
        from ..main_agent import PULSEPAL_SYSTEM_PROMPT
        from ..mri_expert_agent import MRI_EXPERT_SYSTEM_PROMPT
        
        # Pulsepal should handle programming
        assert "programming" in PULSEPAL_SYSTEM_PROMPT.lower()
        assert "code" in PULSEPAL_SYSTEM_PROMPT.lower()
        
        # MRI Expert should handle physics education
        assert "educational" in MRI_EXPERT_SYSTEM_PROMPT.lower()
        assert "physics" in MRI_EXPERT_SYSTEM_PROMPT.lower()
        
        # MRI Expert should NOT handle programming directly
        assert "programming" not in MRI_EXPERT_SYSTEM_PROMPT.lower()


class TestErrorHandling:
    """Test error handling and graceful degradation."""
    
    @pytest.mark.asyncio
    async def test_agent_error_handling(self):
        """Test agent error handling with malformed inputs."""
        test_agent = pulsepal_agent.override(model=TestModel("Error occurred"))
        
        deps = PulsePalDependencies(
            conversation_context=ConversationContext(session_id="error_test")
        )
        
        # Test with empty query
        result = await test_agent.run("", deps=deps)
        assert result.data is not None  # Should handle gracefully
    
    @pytest.mark.asyncio
    async def test_mcp_server_failure_graceful_degradation(self):
        """Test graceful degradation when MCP server fails."""
        deps = PulsePalDependencies()
        deps.mcp_server = None  # Simulate no MCP server
        
        # Should return False but not crash
        result = await deps.ensure_mcp_connection()
        assert result is False
    
    def test_settings_loading_error_messages(self):
        """Test helpful error messages for configuration issues."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                load_settings()
            
            error_message = str(exc_info.value)
            assert "GOOGLE_API_KEY" in error_message or "google_api_key" in error_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])