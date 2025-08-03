"""
Tools for Pulsepal agent including RAG queries and web search.

Provides comprehensive tool integration with native RAG capabilities and web search
with proper error handling.
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any
from pydantic_ai import RunContext
from pydantic import BaseModel, Field

from .dependencies import PulsePalDependencies
from .settings import get_settings
from .rag_service import get_rag_service
from .web_search import get_web_search_service

# Get agent reference - imported after agent creation
def get_agent():
    from .main_agent import pulsepal_agent
    return pulsepal_agent

logger = logging.getLogger(__name__)


class RAGQueryParams(BaseModel):
    """Parameters for RAG query validation."""
    query: str = Field(..., description="Search query for RAG system")
    source: Optional[str] = Field(None, description="Optional source filter")
    match_count: int = Field(5, description="Number of results to return", ge=1, le=20)


class CodeSearchParams(BaseModel):
    """Parameters for code example search validation."""
    query: str = Field(..., description="Code search query")
    source_id: Optional[str] = Field(None, description="Optional source ID filter")
    match_count: int = Field(5, description="Number of results to return", ge=1, le=20)




@get_agent().tool
async def perform_rag_query(
    ctx: RunContext[PulsePalDependencies],
    query: str,
    source: Optional[str] = None,
    match_count: int = 5
) -> str:
    """
    Search the RAG database for Pulseq documentation and information.
    
    Use this tool to find relevant documentation, tutorials, and explanations
    from the comprehensive Pulseq knowledge base.
    
    Args:
        query: Search query for finding relevant documentation
        source: Optional source filter (e.g., 'pulseq.github.io')
        match_count: Number of results to return (1-20)
    
    Returns:
        str: Formatted search results with source information
    """
    try:
        # Validate parameters
        params = RAGQueryParams(query=query, source=source, match_count=match_count)
        
        # Get RAG service
        rag_service = get_rag_service()
        
        # Perform RAG query
        results = rag_service.perform_rag_query(
            query=params.query,
            source=params.source,
            match_count=params.match_count,
            use_hybrid=True  # Use hybrid search by default
        )
        
        logger.info(f"RAG query completed for: {params.query}")
        return results
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        # Try web search as fallback
        try:
            web_service = get_web_search_service()
            web_results = web_service.search_mri_information(query, max_results=match_count)
            return f"RAG database unavailable, showing web search results:\n\n{web_results}"
        except:
            return f"I encountered an error searching the documentation. Please try rephrasing your query or check back later."


@get_agent().tool  
async def search_code_examples(
    ctx: RunContext[PulsePalDependencies],
    query: str,
    source_id: Optional[str] = None,
    match_count: int = 5
) -> str:
    """
    Search for Pulseq code examples and implementations.
    
    Use this tool to find specific code examples, functions, and implementations
    across MATLAB, Octave, and Python Pulseq codebases.
    
    Args:
        query: Code search query (e.g., "spin echo sequence", "gradient echo")
        source_id: Optional source ID filter (e.g., 'pulseq')
        match_count: Number of results to return (1-20)
    
    Returns:
        str: Formatted code examples with descriptions and sources
    """
    try:
        # Validate parameters
        params = CodeSearchParams(query=query, source_id=source_id, match_count=match_count)
        
        # Get RAG service
        rag_service = get_rag_service()
        
        # Search for code examples
        results = rag_service.search_code_examples(
            query=params.query,
            source_id=params.source_id,
            match_count=params.match_count,
            use_hybrid=True  # Use hybrid search by default
        )
        
        logger.info(f"Code search completed for: {params.query}")
        return results
        
    except Exception as e:
        logger.error(f"Code search failed: {e}")
        # Try web search for Pulseq resources as fallback
        try:
            web_service = get_web_search_service()
            web_results = web_service.search_pulseq_resources(query, resource_type="example")
            return f"Code database unavailable, showing web resources:\n\n{web_results}"
        except:
            return f"I encountered an error searching for code examples. Please try a different search query or ask me to generate code directly."


@get_agent().tool
async def get_available_sources(ctx: RunContext[PulsePalDependencies]) -> str:
    """
    Get list of available documentation sources in the RAG database.
    
    Use this tool to discover what documentation sources and repositories
    are available for searching and reference.
    
    Returns:
        str: Formatted list of available sources with descriptions
    """
    try:
        # Get RAG service
        rag_service = get_rag_service()
        
        # Get available sources
        results = rag_service.get_available_sources()
        
        logger.info("Available sources retrieved successfully")
        return results
        
    except Exception as e:
        logger.error(f"Sources query failed: {e}")
        # Return a helpful static list as fallback
        return """## Available Documentation Sources (Cached)

### MATLAB/Octave Sources:
- **github.com/pulseq/pulseq**: Main Pulseq repository with MATLAB source code
- **github.com/pulseq/tutorials**: Tutorial examples and getting started guides
- **github.com/HarmonizedMRI/Pulseq-diffusion**: Diffusion sequence implementations
- **github.com/HarmonizedMRI/Functional**: fMRI sequences and protocols

### Python Sources:
- **github.com/imr-framework/pypulseq**: Python implementation of Pulseq
- **pypulseq.readthedocs.io**: Python Pulseq documentation and API reference

### Documentation Sources:
- **pulseq.github.io**: Official Pulseq documentation and guides
- **harmonizedmri.github.io**: Community-driven Pulseq sequence collection

### Specialized Applications:
- **github.com/HarmonizedMRI/SMS-EPI**: Simultaneous Multi-Slice EPI sequences
- **github.com/pulseq/MR-Physics-with-Pulseq**: Educational MR physics materials

*Note: Unable to connect to live database. Showing cached source list.*"""


