"""
Simplified tools for Pulsepal agent with modern RAG service v2.

Provides tools that use the simplified retrieval-only RAG service,
leaving all intelligence to the LLM.
"""

import logging
import re
from typing import List
from pydantic_ai import RunContext

from .dependencies import PulsePalDependencies
from .rag_service_v2 import ModernPulseqRAG, RetrievalHint
from .web_search import get_web_search_service
from .conversation_logger import get_conversation_logger

# Agent will be set by main_agent.py after creation
pulsepal_agent = None

logger = logging.getLogger(__name__)
conversation_logger = get_conversation_logger()


def extract_functions_from_text(text: str) -> List[str]:
    """
    Simple extraction of function names from text.
    No intelligence, just pattern matching.
    """
    functions = []

    # Patterns for function calls
    patterns = [
        r"(mr\.\w+)",
        r"(seq\.\w+)",
        r"(tra\.\w+)",
        r"(eve\.\w+)",
        r"(opt\.\w+)",
        r"(mr\.aux\.\w+)",
        r"(mr\.aux\.quat\.\w+)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        functions.extend(matches)

    # Remove duplicates while preserving order
    seen = set()
    unique_functions = []
    for func in functions:
        if func not in seen:
            seen.add(func)
            unique_functions.append(func)

    return unique_functions


async def search_pulseq_knowledge(
    ctx: RunContext[PulsePalDependencies], 
    query: str, 
    limit: int = 30,
    forced: bool = False  # NEW parameter for semantic routing
) -> str:
    """
    Search Pulseq knowledge base using modern RAG service.
    Simple retrieval, no classification or pattern matching.

    Args:
        query: Search query
        limit: Maximum number of results
        forced: If True, this search was mandated by semantic router

    Returns:
        Formatted search results
    """
    # Check if this search was forced by semantic router
    if not forced and hasattr(ctx.deps, 'force_rag'):
        forced = ctx.deps.force_rag
    
    # Check if we should skip RAG (pure physics)
    if hasattr(ctx.deps, 'skip_rag') and ctx.deps.skip_rag:
        logger.info("Skipping RAG search for pure physics question")
        return "[Documentation search skipped - pure physics question detected]"
    
    # Get or create RAG service
    if not hasattr(ctx.deps, "rag_v2"):
        ctx.deps.rag_v2 = ModernPulseqRAG()

    # Extract any functions mentioned in the query
    functions = extract_functions_from_text(query)

    # Check if code is provided (simple heuristic)
    code_provided = any(
        pattern in query
        for pattern in ["```", "def ", "function ", "=", ";", "{", "}"]
    )

    # Add forced search hints if available
    search_terms = None
    if hasattr(ctx.deps, 'forced_search_hints') and ctx.deps.forced_search_hints:
        search_terms = ctx.deps.forced_search_hints
    
    # Create hint for retrieval
    hint = RetrievalHint(
        functions_mentioned=functions,
        code_provided=code_provided,
        search_terms=search_terms,
    )

    # Simple retrieval
    results = await ctx.deps.rag_v2.retrieve(query, hint, limit)

    # Format results for display
    if not results["documents"]:
        if forced:
            return """No documentation found, but this query was identified as requiring Pulseq-specific information.
            Please provide guidance based on Pulseq best practices and validate any function names."""
        return "No relevant documentation found in the knowledge base."

    # Add header if this was a forced search
    formatted = []
    if forced:
        formatted.append("[📚 REQUIRED DOCUMENTATION - Query identified as Pulseq-specific]\n")
    for doc in results["documents"][:10]:  # Limit display to top 10
        parts = []

        # Add title/function name
        if doc.get("function_name"):
            parts.append(f"**Function: {doc['function_name']}**")
        elif doc.get("title"):
            parts.append(f"**{doc['title']}**")

        # Add source info
        if doc.get("url"):
            parts.append(f"Source: {doc['url']}")

        # Add content preview
        content = doc.get("content", "")
        if content:
            # Limit content length for display
            if len(content) > 500:
                content = content[:500] + "..."
            parts.append(content)

        # Add similarity score
        if doc.get("similarity"):
            parts.append(f"*Relevance: {doc['similarity']:.2f}*")

        formatted.append("\n".join(parts))

    # Log search event
    if ctx.deps and hasattr(ctx.deps, "conversation_context"):
        # Include forced flag in metadata
        metadata = {
            "functions": functions, 
            "code_provided": code_provided,
            "forced": forced
        }
        if forced:
            metadata["reason"] = "semantic_routing"
            
        conversation_logger.log_search_event(
            ctx.deps.conversation_context.session_id,
            "forced_rag" if forced else "modern_rag",
            query,
            len(results["documents"]),
            metadata,
        )

    return "\n\n---\n\n".join(formatted)


async def verify_function_namespace(
    ctx: RunContext[PulsePalDependencies], function_call: str
) -> str:
    """
    Check if a function call uses the correct namespace.
    This is deterministic validation to prevent common errors.

    Args:
        function_call: Function call to check (e.g., "mr.write")

    Returns:
        Validation result with correction if needed
    """
    # Get or create RAG service
    if not hasattr(ctx.deps, "rag_v2"):
        ctx.deps.rag_v2 = ModernPulseqRAG()

    # Check namespace (synchronous call, no await)
    result = ctx.deps.rag_v2.check_function_namespace(function_call)

    if result["is_error"]:
        return (
            f"❌ **Namespace Error**\n"
            f"Incorrect: `{function_call}`\n"
            f"Correct: `{result['correct_form']}`\n"
            f"Explanation: {result['explanation']}"
        )
    else:
        return f"✅ `{function_call}` uses the correct namespace."


async def validate_pulseq_function(
    ctx: RunContext[PulsePalDependencies], function_name: str
) -> str:
    """
    Validate a Pulseq function name and get corrections if needed.
    Checks against function index and known hallucinations.
    
    Args:
        function_name: Function to validate (e.g., 'seq.calcKspace')
    
    Returns:
        Validation result with corrections if needed
    """
    # Get or create RAG service
    if not hasattr(ctx.deps, "rag_v2"):
        ctx.deps.rag_v2 = ModernPulseqRAG()
    
    result = ctx.deps.rag_v2.validate_function(function_name)
    
    # Format response
    if result["is_valid"]:
        return f"✅ {function_name} is valid"
    else:
        response = [f"❌ {function_name} is not valid"]
        
        if result.get("correct_form"):
            response.append(f"✅ Use: {result['correct_form']}")
        
        if result.get("explanation"):
            response.append(f"ℹ️ {result['explanation']}")
        
        if result.get("suggestions"):
            response.append(f"💡 Suggestions: {', '.join(result['suggestions'])}")
        
        return "\n".join(response)


async def search_web_for_mri_info(
    ctx: RunContext[PulsePalDependencies], query: str
) -> str:
    """
    Search the web for MRI-related information.
    Used when information is not in the Pulseq knowledge base.

    Args:
        query: Search query

    Returns:
        Web search results
    """
    web_search = get_web_search_service()
    if not web_search:
        return "Web search is not configured. Please use the knowledge base search instead."

    # Add MRI/Pulseq context to query if not present
    query_lower = query.lower()
    if not any(
        term in query_lower for term in ["mri", "pulseq", "magnetic", "resonance"]
    ):
        query = f"MRI Pulseq {query}"

    results = await web_search.search(query)

    if not results:
        return "No web results found."

    # Format results
    formatted = []
    for result in results[:5]:  # Top 5 results
        formatted.append(
            f"**{result.get('title', 'Untitled')}**\n"
            f"URL: {result.get('url', 'N/A')}\n"
            f"{result.get('snippet', 'No description available')}"
        )

    return "\n\n---\n\n".join(formatted)
