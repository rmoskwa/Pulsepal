"""
Environment-based configuration for Pulsepal multi-agent system.

Follows main_agent_reference patterns with Pydantic-settings integration
and secure environment variable management.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8", 
        case_sensitive=False,
        extra="ignore"
    )
    
    # LLM Configuration (Gemini)
    google_api_key: str = Field(
        ..., 
        description="Google API key for Gemini model access",
        alias="GOOGLE_API_KEY"
    )
    llm_model: str = Field(
        default="gemini-2.0-flash-exp",
        description="Gemini model name to use",
        alias="LLM_MODEL" 
    )
    
    # Supabase Configuration for MCP Server
    supabase_url: str = Field(
        ...,
        description="Supabase project URL for RAG database",
        alias="SUPABASE_URL"
    )
    supabase_key: str = Field(
        ...,
        description="Supabase anon/service role key",
        alias="SUPABASE_KEY"
    )
    
    # Session Configuration
    max_session_duration_hours: int = Field(
        default=24,
        description="Maximum session duration in hours"
    )
    max_conversation_history: int = Field(
        default=100,
        description="Maximum conversation history entries per session"
    )
    max_code_examples: int = Field(
        default=50,
        description="Maximum code examples stored per session"
    )
    
    # MCP Server Configuration
    mcp_server_timeout: int = Field(
        default=30,
        description="MCP server request timeout in seconds"
    )
    mcp_retry_attempts: int = Field(
        default=3,
        description="Number of retry attempts for MCP server calls"
    )
    
    # RAG Configuration
    default_match_count: int = Field(
        default=5,
        description="Default number of RAG results to return"
    )
    max_match_count: int = Field(
        default=20,
        description="Maximum number of RAG results allowed"
    )


def load_settings() -> Settings:
    """Load settings with proper error handling and environment loading."""
    # Load environment variables from .env file
    load_dotenv()
    
    try:
        settings = Settings()
        logger.info("Settings loaded successfully")
        return settings
        
    except Exception as e:
        error_msg = f"Failed to load settings: {e}"
        
        # Provide helpful error messages for common issues
        if "google_api_key" in str(e).lower():
            error_msg += "\nMake sure to set GOOGLE_API_KEY in your .env file"
        if "supabase_url" in str(e).lower():
            error_msg += "\nMake sure to set SUPABASE_URL in your .env file"
        if "supabase_key" in str(e).lower():
            error_msg += "\nMake sure to set SUPABASE_KEY in your .env file"
            
        logger.error(error_msg)
        raise ValueError(error_msg) from e


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get cached settings instance."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings