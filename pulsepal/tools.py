"""
Simplified tools for Pulsepal agent with intelligent search integration.

Provides a unified tool that intelligently decides when to search the knowledge base
and seamlessly integrates results into responses.

ENHANCED VERSION: Better detection of sequence names and code requests.
"""

import logging
from typing import Optional, List, Dict, Any
from pydantic_ai import RunContext
from pydantic import BaseModel, Field

from .dependencies import PulsePalDependencies
from .settings import get_settings
from .rag_service import get_rag_service
from .web_search import get_web_search_service
from .conversation_logger import get_conversation_logger

# Get agent reference - imported after agent creation
def get_agent():
    from .main_agent import pulsepal_agent
    return pulsepal_agent

logger = logging.getLogger(__name__)
conversation_logger = get_conversation_logger()


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


def enhance_sequence_query(query: str) -> str:
    """
    Enhance queries about sequences to improve search results.
    
    Args:
        query: Original query
        
    Returns:
        Enhanced query for better results
    """
    query_lower = query.lower()
    
    # Common sequence name mappings
    sequence_mappings = {
        'epi': 'echo planar imaging EPI sequence implementation',
        'spin echo': 'spin echo sequence implementation',
        'gradient echo': 'gradient echo GRE sequence implementation', 
        'flash': 'FLASH gradient echo sequence',
        'tse': 'turbo spin echo TSE sequence',
        'fse': 'fast spin echo FSE sequence',
        'mprage': 'MPRAGE sequence implementation',
        'diffusion': 'diffusion weighted imaging sequence',
        # UTE support
        'ute': 'UTE ultra short echo time sequence implementation writeUTE',
        'ultrashort': 'UTE ultra short echo time sequence implementation',
        'zero te': 'zero TE ZTE sequence implementation',
        'zute': 'ZUTE zero ultra short echo time sequence'
    }
    
    # Check if query is asking for a sequence
    for seq_name, enhanced in sequence_mappings.items():
        if seq_name in query_lower:
            # If it's just the sequence name or with "script/example/demo"
            # Increased to 6 to handle "matlab UTE sequence in Pulseq"
            if len(query.split()) <= 6:
                return enhanced
            break
    
    # Add "implementation" if asking for script/demo but not already there
    if any(term in query_lower for term in ['script', 'demo', 'code']) and \
       'implementation' not in query_lower:
        return f"{query} implementation"
    
    return query


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
            
            # ENHANCED: Common MRI sequence names - prioritize code search
            sequence_indicators = [
                'epi', 'echo planar', 'spin echo', 'gradient echo', 'gre',
                'flash', 'tse', 'turbo spin', 'fse', 'fast spin',
                'mprage', 'diffusion', 'dwi', 'dti', 'bold', 'fmri',
                'spiral', 'radial', 'propeller', 'blade', 'fisp', 'ssfp',
                'ir', 'inversion recovery', 'stir', 'flair', 'tof', 'pc',
                # UTE variants
                'ute', 'ultra short', 'ultrashort', 'zero te', 'zte', 'zute'
            ]
            
            # Code/script request indicators
            code_request_indicators = [
                'script', 'demo', 'code', 'example', 'implement', 'sample',
                'show me', 'give me', 'provide', 'write', 'create'
            ]
            
            # Pulseq-specific function indicators
            function_indicators = [
                'mr.', 'makeblock', 'makegauss', 'makearb', 'addblock', 
                'calcgradient', 'calcduration', 'sequence.', 'pypulseq',
                'makesinc', 'maketrapezoid', 'makeadc', '.seq'
            ]
            
            # Check if asking for sequence example/script
            has_sequence = any(seq in query_lower for seq in sequence_indicators)
            has_code_request = any(req in query_lower for req in code_request_indicators)
            has_function = any(func in query_lower for func in function_indicators)
            
            # Prioritize code search for sequence requests
            if (has_sequence and has_code_request) or has_function or \
               ('pulseq' in query_lower and any(term in query_lower for term in ['example', 'script', 'demo'])):
                search_type = "code"
                # Enhance the query for better results
                query = enhance_sequence_query(params.query)
                logger.info(f"Enhanced query from '{params.query}' to '{query}'")
            
            # Documentation indicators (only if not already code)
            elif any(indicator in query_lower for indicator in [
                'tutorial', 'guide', 'documentation', 'manual', 'reference'
            ]):
                search_type = "documentation"
            
            # Source discovery
            elif any(indicator in query_lower for indicator in [
                'available', 'sources', 'repositories', 'what sources', 'list'
            ]):
                search_type = "sources"
            
            # Default: if mentions sequence but not clear, try code first
            elif has_sequence:
                search_type = "code"
                query = enhance_sequence_query(params.query)
            else:
                search_type = "documentation"  # Default fallback
        
        # Execute appropriate search
        if search_type == "sources":
            results = rag_service.get_available_sources()
            logger.info("Retrieved available sources")
            
            # Log search event for debugging
            if ctx.deps and hasattr(ctx.deps, 'conversation_context'):
                conversation_logger.log_search_event(
                    ctx.deps.conversation_context.session_id,
                    "sources",
                    query,
                    results.count("**") if results else 0,
                    {"search_type": search_type}
                )
            
        elif search_type == "code":
            results = rag_service.search_code_examples(
                query=query,  # Use potentially enhanced query
                match_count=params.match_count,
                use_hybrid=True
            )
            logger.info(f"Code search completed for: {query}")
            
            # Log search event for debugging
            if ctx.deps and hasattr(ctx.deps, 'conversation_context'):
                results_count = 0 if "No code examples found" in results else results.count("###")
                conversation_logger.log_search_event(
                    ctx.deps.conversation_context.session_id,
                    "code",
                    query,
                    results_count,
                    {"search_type": search_type, "enhanced_query": query != params.query}
                )
            
            # If no results, try with alternative query
            if "No code examples found" in results and params.query != query:
                # Try original query if we enhanced it
                results = rag_service.search_code_examples(
                    query=params.query,
                    match_count=params.match_count,
                    use_hybrid=True
                )
                logger.info(f"Fallback search with original query: {params.query}")
            
        else:  # documentation or auto fallback
            results = rag_service.perform_rag_query(
                query=params.query,
                match_count=params.match_count,
                use_hybrid=True
            )
            logger.info(f"Documentation search completed for: {params.query}")
            
            # Log search event for debugging
            if ctx.deps and hasattr(ctx.deps, 'conversation_context'):
                results_count = 0 if "No documentation found" in results else results.count("###")
                conversation_logger.log_search_event(
                    ctx.deps.conversation_context.session_id,
                    "documentation",
                    params.query,
                    results_count,
                    {"search_type": search_type}
                )
        
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