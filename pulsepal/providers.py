"""
Model provider configuration for Pulsepal multi-agent system.

Provides Gemini model configuration with proper error handling
following main_agent_reference patterns.
"""

from pydantic_ai.models.gemini import GeminiModel
from .settings import get_settings
import logging
import os

logger = logging.getLogger(__name__)


def get_llm_model() -> GeminiModel:
    """
    Get configured Gemini model with proper environment loading.
    
    Returns:
        GeminiModel: Configured Gemini model instance
        
    Raises:
        ValueError: If configuration is invalid or API key is missing
    """
    settings = get_settings()
    
    try:
        # Set GEMINI_API_KEY environment variable for pydantic-ai
        os.environ['GEMINI_API_KEY'] = settings.google_api_key
        
        # Create Gemini model - it will use GEMINI_API_KEY from environment
        model = GeminiModel(settings.llm_model)
        
        logger.info(f"Initialized Gemini model: {settings.llm_model}")
        return model
        
    except Exception as e:
        error_msg = f"Failed to initialize Gemini model: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def get_test_model() -> str:
    """
    Get test model name for TestModel validation.
    
    Returns:
        str: Model name for testing without API calls
    """
    return "test"


def get_function_model_config() -> dict:
    """
    Get configuration for FunctionModel testing.
    
    Returns:
        dict: Configuration for custom test behavior
    """
    return {
        "function_name": "test_function",
        "description": "Test function for agent validation"
    }