"""
Simplified tools for Pulsepal agent with intelligent search integration.

Provides a unified tool that intelligently decides when to search the knowledge base
and seamlessly integrates results into responses.

ENHANCED VERSION: Better detection of sequence names and code requests.
"""

import logging
import asyncio
from typing import Optional
from pydantic_ai import RunContext
from pydantic import BaseModel, Field

from .dependencies import PulsePalDependencies
from .rag_service import get_rag_service
from .web_search import get_web_search_service
from .conversation_logger import get_conversation_logger
from .timeout_utils import async_timeout

# Agent will be set by main_agent.py after creation
pulsepal_agent = None

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


async def search_pulseq_functions_fast(
    ctx: RunContext[PulsePalDependencies],
    query: str,
    limit: int = 10
) -> str:
    """
    Phase 1: Fast function discovery for immediate display.
    Returns lightweight results in <50ms.
    Used for Level 2 responses (key components).
    """
    try:
        rag_service = get_rag_service()
        results = await rag_service.search_functions_fast(query, limit)
        
        if not results:
            return f"No functions found matching '{query}'. Try a different search term."
        
        # Format lightweight results for Level 2 response
        output = f"## Function Search Results for: '{query}'\n\n"
        for func in results[:5]:  # Show top 5
            output += f"### {func['name']}\n"
            output += f"**Signature**: `{func['signature']}`\n"
            output += f"**Description**: {func['description']}\n"
            if func.get('is_class_method'):
                output += f"**Usage**: `seq.{func['name']}(...)`\n"
            else:
                output += f"**Usage**: `{func.get('calling_pattern', func['name'] + '(...)')}`\n"
            output += "\n"
        
        if len(results) > 5:
            output += f"\n*{len(results)-5} more results available. Use get_function_details for complete parameters.*\n"
            
        return output
        
    except Exception as e:
        logger.error(f"Fast function search failed: {e}")
        return "Function search temporarily unavailable. Try again or use search_pulseq_functions."

async def get_function_details(
    ctx: RunContext[PulsePalDependencies],
    function_names: str | list[str]
) -> str:
    """
    Phase 2: Get complete function details for code generation.
    Use after search_pulseq_functions_fast to get full parameters.
    Used for Level 3 responses (complete implementation).
    """
    try:
        # Handle single function or list
        if isinstance(function_names, str):
            function_names = [function_names]
            
        rag_service = get_rag_service()
        details = await rag_service.get_function_details(function_names)
        
        if not details:
            return f"No details found for functions: {', '.join(function_names)}"
        
        # Format detailed results for Level 3 implementation
        output = "## Complete Function Details\n\n"
        for func in details:
            output += f"### {func['name']}\n"
            output += f"**Full Signature**: `{func['signature']}`\n"
            if func.get('parameters'):
                output += f"**Parameters**:\n{func['parameters']}\n"
            if func.get('usage_examples'):
                output += f"**Examples**:\n```matlab\n{func['usage_examples']}\n```\n"
            if func.get('returns'):
                output += f"**Returns**: {func['returns']}\n"
            output += "\n---\n"
            
        return output
        
    except Exception as e:
        logger.error(f"Function details fetch failed: {e}")
        return "Could not retrieve function details. Use search_pulseq_functions for basic info."

async def get_official_sequence_example(
    ctx: RunContext[PulsePalDependencies],
    sequence_type: str
) -> str:
    """
    Get official, validated sequence example from Pulseq demoSeq.
    These are tested, working examples for Pulseq v1.5.0.
    Used for Level 3 responses (complete implementation).
    
    Available types: EPI, SpinEcho, GradientEcho, TSE, MPRAGE, UTE, HASTE, TrueFISP, PRESS, Spiral
    """
    try:
        rag_service = get_rag_service()
        
        # Use timeout to prevent long-running queries
        import asyncio
        result = await asyncio.wait_for(
            rag_service.get_official_sequence(sequence_type),
            timeout=5.0  # 5 second timeout
        )
        
        if not result:
            return (f"No official example found for '{sequence_type}'.\n"
                   f"Available sequences: EPI, SpinEcho, GradientEcho, TSE, MPRAGE, "
                   f"UTE, HASTE, TrueFISP, PRESS, Spiral\n\n"
                   f"Try search_pulseq_knowledge for community examples.")
        
        # Parse content if it contains the summary separator
        content = result.get('content', '')
        if not content:
            # If content is missing, it means we only got the summary
            # This happens when the tool fetches the wrong fields
            logger.warning(f"No content in result for {sequence_type}, result keys: {result.keys()}")
            # Try to use ai_summary if available
            if 'ai_summary' in result and result['ai_summary']:
                return (f"## {sequence_type} Sequence Summary\n\n"
                       f"{result['ai_summary']}\n\n"
                       f"*Note: Full implementation code not available. "
                       f"Try search_pulseq_knowledge for complete examples.*")
            return (f"Found {sequence_type} but content is missing.\n"
                   f"Try search_pulseq_knowledge for community examples.")
        if '---' in content:
            # Extract just the code part after the summary
            parts = content.split('---', 1)
            if len(parts) > 1:
                content = parts[1].strip()
        
        # Return the official example directly for Level 3
        output = f"## Official Pulseq Example: {result['sequence_type']}\n"
        output += f"*Source: {result['file_name']}*\n\n"
        output += "```matlab\n"
        output += content[:10000]  # Limit to 10k chars
        if len(content) > 10000:
            output += "\n\n% ... [truncated for display]"
        output += "\n```\n"
        
        return output
        
    except asyncio.TimeoutError:
        logger.warning(f"Official sequence fetch timed out for {sequence_type}")
        return (f"Request timed out. The '{sequence_type}' sequence may be too large.\n"
                f"Try search_pulseq_knowledge for a more targeted search.")
    except Exception as e:
        logger.error(f"Official sequence fetch failed: {e}")
        # Fallback to broader search
        return await search_pulseq_knowledge(ctx, f"{sequence_type} sequence example", search_type="code")

# Removed timeout temporarily to debug
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
    
    When search_type="auto":
    - Classifies intent
    - Routes to appropriate search
    - Formats adaptively
    
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
        
        # Use enhanced perform_rag_query for auto routing
        if params.search_type == "auto":
            # Check if async method exists, otherwise use sync version
            if hasattr(rag_service, 'perform_rag_query') and asyncio.iscoroutinefunction(rag_service.perform_rag_query):
                results = await rag_service.perform_rag_query(
                    query=params.query,
                    search_type="auto",
                    match_count=params.match_count
                )
            else:
                # Call the enhanced classification-based search
                # Create async wrapper if needed
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda: rag_service.perform_rag_query(
                        query=params.query,
                        match_count=params.match_count,
                        use_hybrid=True
                    )
                )
            
            # Log search event for debugging
            if ctx.deps and hasattr(ctx.deps, 'conversation_context'):
                conversation_logger.log_search_event(
                    ctx.deps.conversation_context.session_id,
                    "enhanced_auto",
                    params.query,
                    1 if results else 0,
                    {"search_type": "auto", "enhanced": True}
                )
            
            return results
        
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
            # First check for official sequences if it's a sequence request
            if any(seq in query.lower() for seq in ['epi', 'spin echo', 'gradient echo', 'tse', 'mprage', 'ute', 'haste', 'trufi', 'press', 'spiral']):
                # Use async version of search_code_examples if available
                if hasattr(rag_service, 'search_code_examples') and asyncio.iscoroutinefunction(rag_service.search_code_examples):
                    results = await rag_service.search_code_examples(
                        query=query,
                        match_count=params.match_count
                    )
                else:
                    loop = asyncio.get_event_loop()
                    results = await loop.run_in_executor(
                        None,
                        lambda: rag_service.search_code_examples(
                            query=query,
                            match_count=params.match_count
                        )
                    )
                # Log and return early - we already have formatted results
                logger.info(f"Code search completed for sequence: {query}")
                if ctx.deps and hasattr(ctx.deps, 'conversation_context'):
                    conversation_logger.log_search_event(
                        ctx.deps.conversation_context.session_id,
                        "code",
                        query,
                        1 if results and "No code examples found" not in results else 0,
                        {"search_type": search_type, "sequence_search": True}
                    )
                return results
            else:
                # Use the new smart search strategy
                # Classify the query to determine search approach
                strategy, metadata = rag_service.classify_search_strategy(query)
                
                if strategy == "vector_enhanced":
                    # Enhanced search for MRI sequences
                    # We directly use the enhanced search from RAG service
                    seq_type = metadata.get("sequence_type", "")
                    raw_results = rag_service.supabase_client.perform_hybrid_search(
                        query=query,
                        match_count=50,  # Reduced for better performance
                        search_type="code_examples",
                        keyword_query_override=seq_type
                    )
                    
                    # Apply the new scoring method
                    if raw_results:
                        scored_results = []
                        for result in raw_results:
                            # Use the rag_service's scoring method
                            score = rag_service._score_sequence_relevance(result, seq_type)
                            
                            if score > 0 or len(scored_results) < 10:
                                result['relevance_score'] = score
                                scored_results.append(result)
                        
                        scored_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
                        raw_results = scored_results
                else:
                    # Hybrid search with optional filtering
                    filtered_query = metadata.get("filtered_query") if strategy == "hybrid_filtered" else None
                    raw_results = rag_service.supabase_client.perform_hybrid_search(
                        query=query,
                        match_count=50,  # Reduced for better performance
                        search_type="code_examples",
                        keyword_query_override=filtered_query
                    )
            
            # Format results - always shows top result directly
            formatted_results, _ = rag_service.format_code_results_interactive(raw_results, query)
            results = formatted_results
            logger.info(f"Code search completed for: {query}")
            
            # Log search event for debugging
            if ctx.deps and hasattr(ctx.deps, 'conversation_context'):
                results_count = len(raw_results) if raw_results else 0
                conversation_logger.log_search_event(
                    ctx.deps.conversation_context.session_id,
                    "code",
                    query,
                    results_count,
                    {"search_type": search_type, "enhanced_query": query != params.query}
                )
            
            # If no results and we enhanced the query, be transparent about trying alternatives
            if "No code examples found" in results and params.query != query:
                # Be transparent about the search enhancement
                transparency_msg = f"\n*Note: No exact matches for '{params.query}'. "
                transparency_msg += f"I searched for related term: '{query}'*\n\n"
                
                # Try original query if we enhanced it
                # Use the same smart strategy for the fallback
                fallback_strategy, fallback_metadata = rag_service.classify_search_strategy(params.query)
                if fallback_strategy == "vector_enhanced":
                    # Same logic as above for enhanced search
                    seq_type = fallback_metadata.get("sequence_type", "")
                    fallback_raw_results = rag_service.supabase_client.perform_hybrid_search(
                        query=params.query,
                        match_count=50,  # Reduced for better performance
                        search_type="code_examples",
                        keyword_query_override=seq_type
                    )
                    # Apply the new scoring method
                    if fallback_raw_results:
                        scored = []
                        for r in fallback_raw_results:
                            score = rag_service._score_sequence_relevance(r, seq_type)
                            if score > 0 or len(scored) < 10:
                                r['relevance_score'] = score
                                scored.append(r)
                        scored.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
                        fallback_raw_results = scored
                else:
                    fallback_filtered = fallback_metadata.get("filtered_query") if fallback_strategy == "hybrid_filtered" else None
                    fallback_raw_results = rag_service.supabase_client.perform_hybrid_search(
                        query=params.query,
                        match_count=50,  # Reduced for better performance
                        search_type="code_examples",
                        keyword_query_override=fallback_filtered
                    )
                logger.info(f"Fallback search with original query: {params.query}")
                
                if fallback_raw_results:
                    fallback_formatted, _ = rag_service.format_code_results_interactive(fallback_raw_results, params.query)
                    results = transparency_msg + fallback_formatted
                else:
                    # Both searches failed - provide helpful message
                    results = f"## No code examples found for: '{params.query}'\n\n"
                    results += f"*I also searched for the enhanced term: '{query}'*\n\n"
                    results += "This might be because:\n"
                    results += "1. The function/sequence hasn't been implemented in the examples\n"
                    results += "2. It uses different terminology in the codebase\n"
                    results += "3. It might be a method of a class (e.g., seq.methodName())\n\n"
                    results += "Would you like me to search for similar concepts or explain the theory?"
            
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


@async_timeout(seconds=8)  # 8 second timeout for API searches
async def search_pulseq_functions(
    ctx: RunContext[PulsePalDependencies],
    query: str,
    language: Optional[str] = None,
    match_count: int = 5
) -> str:
    """
    Search for Pulseq API functions by name or description.
    
    This tool is specifically designed for finding Pulseq function definitions,
    signatures, parameters, and usage information from the API reference.
    
    Args:
        query: Function name or description to search for
        language: Optional language filter (matlab, python, cpp)
        match_count: Number of results to return (1-10)
    
    Returns:
        str: Formatted API function results with signatures and parameters
    """
    try:
        # Validate parameters
        if match_count < 1 or match_count > 10:
            match_count = 5
        
        # Normalize language filter
        if language:
            language = language.lower()
            if language not in ['matlab', 'python', 'cpp']:
                language = None
        
        # Get RAG service
        rag_service = get_rag_service()
        
        # Search API functions
        results = rag_service.search_api_functions(
            query=query,
            match_count=match_count,
            language_filter=language
        )
        
        logger.info(f"API function search completed for: {query}")
        
        # Log search event for debugging
        if ctx.deps and hasattr(ctx.deps, 'conversation_context'):
            results_count = 0 if "No API functions found" in results else results.count("###")
            conversation_logger.log_search_event(
                ctx.deps.conversation_context.session_id,
                "api_functions",
                query,
                results_count,
                {"language_filter": language, "match_count": match_count}
            )
        
        return results
        
    except Exception as e:
        logger.error(f"Pulseq function search failed: {e}")
        
        # Graceful fallback to general knowledge search
        try:
            rag_service = get_rag_service()
            fallback_results = rag_service.perform_rag_query(
                query=f"Pulseq function {query}",
                match_count=match_count,
                use_hybrid=True
            )
            
            return f"API function search temporarily unavailable. Here are general search results:\n\n{fallback_results}"
            
        except Exception as fallback_error:
            logger.error(f"Fallback search also failed: {fallback_error}")
            
            # Final fallback message
            return f"""I encountered an error searching for the Pulseq function: "{query}". 

This could be due to a temporary database issue. Please try:
1. Rephrasing your query 
2. Using the general search_pulseq_knowledge tool instead
3. Asking me to explain the function concept using my general knowledge"""


# REMOVED DUPLICATE - Using the optimized version at line 166 instead


@async_timeout(seconds=10)  # 10 second timeout for unified searches
async def search_all_pulseq_sources(
    ctx: RunContext[PulsePalDependencies],
    query: str,
    match_count: int = 10
) -> str:
    """
    Search across all Pulseq data sources intelligently based on query type.
    
    This tool automatically classifies your query and searches the most relevant
    sources (API functions, code examples, and/or documentation) based on what
    you're asking for.
    
    Args:
        query: Search query - can be about functions, examples, or concepts
        match_count: Total number of results across all sources (5-20)
    
    Returns:
        str: Intelligently formatted results from all relevant sources
    """
    try:
        # Validate parameters
        if match_count < 5 or match_count > 20:
            match_count = 10
        
        # Get RAG service
        rag_service = get_rag_service()
        
        # Perform unified search across all sources
        results = rag_service.search_all_sources(
            query=query,
            match_count=match_count
        )
        
        logger.info(f"Unified search completed for: {query}")
        
        # Log search event for debugging
        if ctx.deps and hasattr(ctx.deps, 'conversation_context'):
            # Count total results across all sources
            results_count = results.count("###") + results.count("🔧") + results.count("💻") + results.count("📚")
            conversation_logger.log_search_event(
                ctx.deps.conversation_context.session_id,
                "unified",
                query,
                results_count,
                {"match_count": match_count}
            )
        
        return results
        
    except Exception as e:
        logger.error(f"Unified search failed: {e}")
        
        # Graceful fallback to traditional search
        try:
            return await search_pulseq_knowledge(
                ctx, query, search_type="auto", match_count=match_count
            )
            
        except Exception as fallback_error:
            logger.error(f"Fallback search also failed: {fallback_error}")
            
            # Final fallback message
            return f"""I encountered an error performing a comprehensive search for: "{query}". 

This could be due to a temporary database issue. Please try:
1. Using a more specific search tool (search_pulseq_functions for API functions)
2. Breaking down your query into smaller parts
3. Asking me to explain the concept using my general knowledge"""


# Legacy tool aliases for backward compatibility during transition
# These will be removed in a future update

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


async def search_code_examples(
    ctx: RunContext[PulsePalDependencies],
    query: str,
    source_id: Optional[str] = None,
    match_count: int = 5
) -> str:
    """
    DEPRECATED: Use search_pulseq_knowledge with search_type='code' instead.
    Search for Pulseq code examples and implementations.
    NOTE: This searches the code_examples_legacy table which is deprecated.
    """
    logger.warning("search_code_examples is deprecated. Use search_pulseq_knowledge instead.")
    return await search_pulseq_knowledge(
        ctx, query, search_type="code", match_count=match_count
    )


async def get_available_sources(ctx: RunContext[PulsePalDependencies]) -> str:
    """
    Legacy tool: Use search_pulseq_knowledge instead.
    Get list of available documentation sources in the RAG database.
    """
    logger.warning("get_available_sources is deprecated. Use search_pulseq_knowledge instead.")
    return await search_pulseq_knowledge(
        ctx, "available sources", search_type="sources"
    )