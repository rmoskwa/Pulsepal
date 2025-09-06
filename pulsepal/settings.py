"""
Environment-based configuration for Pulsepal multi-agent system.

Follows main_agent_reference patterns with Pydantic-settings integration
and secure environment variable management.
"""

import logging
from typing import Optional

from dotenv import load_dotenv
from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings

# Configure logging - will be reconfigured based on settings
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM Configuration (Gemini)
    google_api_key: str = Field(
        ...,
        description="Google API key for Gemini model access",
        alias="GOOGLE_API_KEY",
    )
    llm_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model name to use",
        alias="LLM_MODEL",
    )

    # Supabase Configuration for RAG
    supabase_url: str = Field(
        ...,
        description="Supabase project URL for RAG database",
        alias="SUPABASE_URL",
    )
    supabase_key: str = Field(
        ...,
        description="Supabase service role key for full access",
        alias="SUPABASE_KEY",
    )

    # Google Embeddings API Configuration
    google_api_key_embedding: str = Field(
        None,
        description="Google API key for embeddings (optional, uses main key if not set)",
        alias="GOOGLE_API_KEY_EMBEDDING",
    )

    # Session Configuration
    max_session_duration_hours: int = Field(
        default=24,
        description="Maximum session duration in hours",
    )
    max_conversation_history: int = Field(
        default=100,
        description="Maximum conversation history entries per session",
    )
    max_code_examples: int = Field(
        default=50,
        description="Maximum code examples stored per session",
    )

    # RAG Configuration
    default_match_count: int = Field(
        default=5,
        description="Default number of RAG results to return",
    )
    max_match_count: int = Field(
        default=20,
        description="Maximum number of RAG results allowed",
    )
    use_hybrid_search: bool = Field(
        default=True,
        description="Use hybrid search (vector + keyword) for better results",
    )
    use_contextual_embeddings: bool = Field(
        default=False,
        description="Use contextual embeddings for document chunks",
    )

    # Hybrid Search Configuration
    hybrid_search_enabled: bool = Field(
        default=True,
        description="Enable hybrid search (BM25 + vector) with RRF fusion and reranking",
        alias="HYBRID_SEARCH_ENABLED",
    )
    hybrid_bm25_k: int = Field(
        default=20,
        description="Number of results to retrieve from BM25 keyword search",
        alias="HYBRID_BM25_K",
    )
    hybrid_vector_k: int = Field(
        default=20,
        description="Number of results to retrieve from vector semantic search",
        alias="HYBRID_VECTOR_K",
    )
    hybrid_rrf_k: int = Field(
        default=60,
        description="RRF k parameter for combining BM25 and vector search results",
        alias="HYBRID_RRF_K",
    )
    hybrid_rerank_top_k: int = Field(
        default=15,
        description="Number of top results to send to neural reranker",
        alias="HYBRID_RERANK_TOP_K",
    )
    hybrid_final_top_k: int = Field(
        default=3,
        description="Number of final reranked results to return to LLM",
        alias="HYBRID_FINAL_TOP_K",
    )

    # Language Configuration
    default_language: str = Field(
        default="matlab",
        description="Default programming language for code examples (matlab/python)",
        alias="DEFAULT_LANGUAGE",
    )

    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        alias="LOG_LEVEL",
    )

    # Alpha Testing Configuration (migrated from alpha_keys.json)
    alpha_api_keys: str = Field(
        default="",
        description="Comma-separated list of alpha testing API keys",
        alias="ALPHA_API_KEYS",
    )
    enable_alpha_auth: bool = Field(
        default=False,
        description="Enable alpha testing authentication",
        alias="ENABLE_ALPHA_AUTH",
    )
    alpha_user_daily_limit: int = Field(
        default=100,
        description="Maximum requests per alpha user per day",
        alias="ALPHA_USER_DAILY_LIMIT",
    )

    # Log Rotation Configuration
    log_retention_days: int = Field(
        default=30,
        description="Number of days to retain session logs",
        alias="LOG_RETENTION_DAYS",
    )
    max_log_size_gb: float = Field(
        default=1.0,
        description="Maximum total size of log directory in GB",
        alias="MAX_LOG_SIZE_GB",
    )
    archive_important_sessions: bool = Field(
        default=True,
        description="Archive important sessions before deletion",
        alias="ARCHIVE_IMPORTANT",
    )
    importance_threshold: float = Field(
        default=0.7,
        description="Score threshold for considering a session important",
        alias="IMPORTANCE_THRESHOLD",
    )
    rotation_check_interval: int = Field(
        default=3600,
        description="Interval in seconds between rotation checks",
        alias="ROTATION_INTERVAL",
    )

    # Reranker Configuration
    reranker_model_path: Optional[str] = Field(
        default=None,  # Will be auto-determined based on platform
        description="Path to store/load reranker model files (defaults to platform-specific)",
        alias="RERANKER_MODEL_PATH",
    )
    reranker_model_name: str = Field(
        default="BAAI/bge-reranker-base",
        description="HuggingFace model identifier for reranker",
        alias="RERANKER_MODEL_NAME",
    )
    reranker_batch_size: int = Field(
        default=15,
        description="Maximum documents to process in reranking batch",
        alias="RERANKER_BATCH_SIZE",
    )
    reranker_timeout: int = Field(
        default=30,
        description="Model initialization timeout in seconds",
        alias="RERANKER_TIMEOUT",
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
