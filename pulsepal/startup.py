"""
Startup initialization module for Pulsepal.

This module handles eager loading of services at application startup
to avoid lazy loading delays during runtime.
"""

import logging
import os
import sys
from typing import Optional

from .embeddings import create_embedding
from .settings import get_settings
from .supabase_client import SupabaseRAGClient

logger = logging.getLogger(__name__)

# Global instances initialized at startup
_supabase_client: Optional[SupabaseRAGClient] = None
_settings = None
_initialization_complete = False


def initialize_supabase() -> SupabaseRAGClient:
    """
    Initialize Supabase client at startup.
    
    Returns:
        SupabaseRAGClient instance
    
    Raises:
        ValueError: If Supabase credentials are not configured
    """
    global _supabase_client
    
    if _supabase_client is not None:
        return _supabase_client
    
    logger.info("Initializing Supabase client at startup...")
    
    try:
        # Get settings
        settings = get_settings()
        
        # Initialize Supabase client with credentials
        _supabase_client = SupabaseRAGClient(
            url=settings.supabase_url,
            key=settings.supabase_key
        )
        
        # Also set it in the supabase_client module's global
        import pulsepal.supabase_client as sc
        sc._supabase_client = _supabase_client
        
        logger.info("✅ Supabase client initialized successfully")
        return _supabase_client
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Supabase client: {e}")
        raise


def initialize_embeddings():
    """
    Pre-initialize the embeddings system to avoid lazy loading delays.
    """
    logger.info("Pre-initializing embeddings system...")
    
    try:
        # Just import and initialize the service without making API calls
        from .embeddings import get_embedding_service, get_embedding_dimensions
        
        # Initialize the service (this creates the provider)
        service = get_embedding_service()
        dimensions = get_embedding_dimensions()
        
        logger.info(f"✅ Embeddings service initialized (dimension: {dimensions})")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize embeddings: {e}")
        # Don't raise - embeddings might still work later


def initialize_all_services():
    """
    Initialize all services at application startup.
    
    This function should be called once at the beginning of the application
    to eagerly load all services and avoid lazy loading delays.
    """
    global _initialization_complete, _settings
    
    if _initialization_complete:
        logger.debug("Services already initialized, skipping...")
        return
    
    logger.info("=" * 60)
    logger.info("PULSEPAL STARTUP INITIALIZATION")
    logger.info("=" * 60)
    
    try:
        # Load settings first
        logger.info("Loading configuration...")
        _settings = get_settings()
        logger.info("✅ Configuration loaded")
        
        # Initialize Supabase client
        supabase_client = initialize_supabase()
        
        # Initialize embeddings
        initialize_embeddings()
        
        # Skip connection test during startup to avoid delays
        # The connection will be tested when first used
        logger.info("✅ Supabase client ready (connection will be verified on first use)")
        
        _initialization_complete = True
        logger.info("=" * 60)
        logger.info("✅ STARTUP INITIALIZATION COMPLETE")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Startup initialization failed: {e}")
        logger.error("The application may still work but with degraded performance")
        # Don't raise - allow the app to continue with lazy loading fallback


def get_initialized_supabase_client() -> Optional[SupabaseRAGClient]:
    """
    Get the pre-initialized Supabase client.
    
    Returns:
        SupabaseRAGClient instance if initialized, None otherwise
    """
    return _supabase_client


def ensure_initialization():
    """
    Ensure services are initialized. Can be called multiple times safely.
    """
    if not _initialization_complete:
        initialize_all_services()