"""
Simplified tools for Pulsepal agent with intelligent search integration.

Provides a unified tool that intelligently decides when to search the knowledge base
and seamlessly integrates results into responses.
"""

import logging
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


class PulseqSearchParams(BaseModel):
    """Parameters for unified Pulseq search validation."""
    query: str = Field(..., description="Search query for Pulseq knowledge base")
    search_type: str = Field(
        default="auto", 
        description="Type of search: 'documentation', 'code', 'sources', or 'auto'"
    )
    match_count: int = Field(5, description="Number of results to return", ge=1, le=20)
    force_search: bool = Field(
        default=False, 
        description="Force search even for general knowledge queries"
    )


@get_agent().tool
async def search_pulseq_knowledge(
    ctx: RunContext[PulsePalDependencies],
    query: str,
    search_type: str = "auto",
    match_count: int = 5,
    force_search: bool = False
) -> str:
    """
    Intelligently search the Pulseq knowledge base when specific implementation details are needed.
    
    This tool is designed to be used selectively - only when the query requires specific Pulseq
    function details, implementation examples, or version-specific information that goes beyond
    general MRI physics and programming knowledge.
    
    Args:
        query: Search query for Pulseq-specific information
        search_type: Type of search ('documentation', 'code', 'sources', or 'auto')
        match_count: Number of results to return (1-20)
        force_search: Force search even for general queries (use sparingly)
    
    Returns:
        str: Formatted search results with source information
    """
    try:
        # Validate parameters
        params = PulseqSearchParams(
            query=query, 
            search_type=search_type, 
            match_count=match_count,
            force_search=force_search
        )
        
        # Get RAG service
        rag_service = get_rag_service()
        
        # Determine search strategy based on type
        if params.search_type == "auto":
            # Intelligent routing based on query content
            query_lower = query.lower()
            
            # Code-specific indicators
            if any(indicator in query_lower for indicator in [
                'mr.', 'makeblock', 'makegauss', 'makearb', 'addblock', 
                'calcgradient', 'calcduration', 'sequence.', 'pypulseq',
                'function', 'parameters', 'syntax', 'implementation'
            ]):
                search_type = "code"
            
            # Documentation indicators  
            elif any(indicator in query_lower for indicator in [
                'molli', 'sms-epi', '3depi', 'diffusion', 'fmri',
                'tutorial', 'example', 'guide', 'documentation'
            ]):
                search_type = "documentation"
            
            # Source discovery
            elif any(indicator in query_lower for indicator in [
                'available', 'sources', 'repositories', 'what', 'list'
            ]):
                search_type = "sources"
            
            else:
                search_type = "documentation"  # Default fallback
        
        # Execute appropriate search
        if search_type == "sources":
            results = rag_service.get_available_sources()
            logger.info("Retrieved available sources")
            
        elif search_type == "code":
            results = rag_service.search_code_examples(
                query=params.query,
                match_count=params.match_count,
                use_hybrid=True
            )
            logger.info(f"Code search completed for: {params.query}")
            
        else:  # documentation or auto fallback
            results = rag_service.perform_rag_query(
                query=params.query,
                match_count=params.match_count,
                use_hybrid=True
            )
            logger.info(f"Documentation search completed for: {params.query}")
        
        return results
        
    except Exception as e:
        logger.error(f"Pulseq knowledge search failed: {e}")
        
        # Graceful fallback to web search
        try:
            web_service = get_web_search_service()
            
            if search_type == "code":
                web_results = web_service.search_pulseq_resources(query, resource_type="example")
            else:
                web_results = web_service.search_mri_information(query, max_results=match_count)
            
            return f"Knowledge base temporarily unavailable. Here are web search results:\n\n{web_results}"
            
        except Exception as web_error:
            logger.error(f"Fallback web search also failed: {web_error}")
            
            # Final fallback message
            return f"""I encountered an error accessing the Pulseq knowledge base for your query: "{query}". 

Since this appears to be a Pulseq-specific question that would benefit from searching our documentation, please try rephrasing your query or ask me to explain the concept using my general knowledge instead."""


# Legacy tool aliases for backward compatibility during transition
# These will be removed in a future update

@get_agent().tool
async def perform_rag_query(
    ctx: RunContext[PulsePalDependencies],
    query: str,
    source: Optional[str] = None,
    match_count: int = 5
) -> str:
    """
    Legacy tool: Use search_pulseq_knowledge instead.
    Search the RAG database for Pulseq documentation and information.
    """
    logger.warning("perform_rag_query is deprecated. Use search_pulseq_knowledge instead.")
    return await search_pulseq_knowledge(
        ctx, query, search_type="documentation", match_count=match_count
    )


@get_agent().tool  
async def search_code_examples(
    ctx: RunContext[PulsePalDependencies],
    query: str,
    source_id: Optional[str] = None,
    match_count: int = 5
) -> str:
    """
    Legacy tool: Use search_pulseq_knowledge instead.
    Search for Pulseq code examples and implementations.
    """
    logger.warning("search_code_examples is deprecated. Use search_pulseq_knowledge instead.")
    return await search_pulseq_knowledge(
        ctx, query, search_type="code", match_count=match_count
    )


@get_agent().tool
async def get_available_sources(ctx: RunContext[PulsePalDependencies]) -> str:
    """
    Legacy tool: Use search_pulseq_knowledge instead.
    Get list of available documentation sources in the RAG database.
    """
    logger.warning("get_available_sources is deprecated. Use search_pulseq_knowledge instead.")
    return await search_pulseq_knowledge(
        ctx, "available sources", search_type="sources"
    )