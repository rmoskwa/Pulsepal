"""
Simplified tools for Pulsepal agent with modern RAG service v2.

Provides tools that use the simplified retrieval-only RAG service,
leaving all intelligence to the LLM.
"""

import logging
import re
from typing import List, Dict, Optional
from pydantic_ai import RunContext

from .dependencies import PulsePalDependencies
from .rag_service_v2 import ModernPulseqRAG
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
    forced: bool = False,  # NEW parameter for semantic routing
    sources: Optional[List[str]] = None,  # NEW parameter for source specification
) -> str:
    """
    Search Pulseq knowledge with intelligent source selection.

    Available sources (choose based on query intent):
    - "api_reference": Function signatures, parameters, return types (best for: specs, syntax)
    - "crawled_pages": Code examples, tutorials, implementations (best for: how-to, debugging)
    - "official_sequence_examples": Complete educational sequences (best for: learning, templates)

    Args:
        query: Search query
        limit: Maximum results (default 30)
        forced: Internal flag for semantic routing
        sources: List of sources to search. Examples:
                 ["api_reference"] for parameter questions
                 ["crawled_pages", "official_sequence_examples"] for learning
                 None to search all sources

    Returns:
        Formatted results with source attribution and synthesis hints
    """
    # Check if this search was forced by semantic router
    if not forced and hasattr(ctx.deps, "force_rag"):
        forced = ctx.deps.force_rag

    # Check if we should skip RAG (pure physics)
    if hasattr(ctx.deps, "skip_rag") and ctx.deps.skip_rag:
        logger.info("Skipping RAG search for pure physics question")
        return "[Documentation search skipped - pure physics question detected]"

    # Get or create RAG service
    if not hasattr(ctx.deps, "rag_v2"):
        ctx.deps.rag_v2 = ModernPulseqRAG()

    # Get source hints from semantic router if available
    source_hints = None
    if hasattr(ctx.deps, "forced_search_hints") and ctx.deps.forced_search_hints:
        source_hints = {"search_terms": ctx.deps.forced_search_hints}

    # Use new source-aware search
    results = await ctx.deps.rag_v2.search_with_source_awareness(
        query=query, sources=sources, forced=forced, source_hints=source_hints
    )

    # Format results for display
    if not results.get("results_by_source"):
        if forced:
            return """No documentation found, but this query was identified as requiring Pulseq-specific information.
            Please provide guidance based on Pulseq best practices and validate any function names."""
        return "No relevant documentation found in the knowledge base."

    # Build formatted response
    formatted = []

    # Add header if this was a forced search
    if forced:
        formatted.append(
            "[📚 REQUIRED DOCUMENTATION - Query identified as Pulseq-specific]\n"
        )

    # Add search metadata
    formatted.append(
        f"**Sources searched:** {', '.join(results['search_metadata']['sources_searched'])}"
    )
    formatted.append(
        f"**Total results:** {results['search_metadata']['total_results']}\n"
    )

    # Add synthesis hints
    if results.get("synthesis_hints"):
        formatted.append("**Key insights:**")
        for hint in results["synthesis_hints"]:
            formatted.append(f"• {hint}")
        formatted.append("")

    # Format results by source
    for source_type, source_results in results.get("results_by_source", {}).items():
        formatted.append(f"\n### {source_type.replace('_', ' ').title()}\n")

        for idx, result in enumerate(source_results[:5], 1):  # Limit to 5 per source
            if source_type == "api_documentation":
                formatted.append(format_api_result(result))
            elif source_type == "examples_and_docs":
                formatted.append(format_example_result(result))
            elif source_type == "tutorials":
                formatted.append(format_tutorial_result(result))

    # Add synthesis recommendations
    if results.get("synthesis_recommendations"):
        formatted.append("\n**Recommendations:**")
        for rec in results["synthesis_recommendations"]:
            formatted.append(f"• {rec}")

    # Log search event
    if ctx.deps and hasattr(ctx.deps, "conversation_context"):
        metadata = {
            "forced": forced,
            "sources_searched": results["search_metadata"]["sources_searched"],
            "total_results": results["search_metadata"]["total_results"],
        }
        if forced:
            metadata["reason"] = "semantic_routing"

        conversation_logger.log_search_event(
            ctx.deps.conversation_context.session_id,
            "source_aware_rag",
            query,
            results["search_metadata"]["total_results"],
            metadata,
        )

    return "\n".join(formatted)


def format_api_result(result: Dict) -> str:
    """Format API documentation result for display."""
    parts = []
    func_info = result.get("function", {})
    tech_details = result.get("technical_details", {})

    parts.append(f"**{func_info.get('name', 'Unknown Function')}**")

    if func_info.get("purpose"):
        parts.append(f"Purpose: {func_info['purpose']}")

    if func_info.get("usage"):
        parts.append(f"Usage: `{func_info['usage']}`")

    if tech_details.get("parameters"):
        parts.append(f"Parameters:\n{tech_details['parameters']}")

    if tech_details.get("returns"):
        parts.append(f"Returns: {tech_details['returns']}")

    return "\n".join(parts) + "\n"


def format_example_result(result: Dict) -> str:
    """Format code example result for display."""
    parts = []

    if result.get("source_type") == "DOCUMENTATION_MULTI_PART":
        doc = result.get("document", {})
        parts.append(
            f"**{doc.get('title', 'Document')}** ({doc.get('total_parts', 1)} parts)"
        )
        parts.append(f"URL: {doc.get('url', 'N/A')}")

        # Show truncated content
        content = doc.get("content", "")
        if len(content) > 500:
            content = content[:500] + "..."
        parts.append(content)
    else:
        content_info = result.get("content", {})
        parts.append(f"**{content_info.get('title', 'Example')}**")

        if content_info.get("url"):
            parts.append(f"URL: {content_info['url']}")

        # Show truncated content
        text = content_info.get("text", "")
        if len(text) > 500:
            text = text[:500] + "..."
        parts.append(text)

    return "\n".join(parts) + "\n"


def format_tutorial_result(result: Dict) -> str:
    """Format tutorial/sequence result for display."""
    parts = []
    tutorial = result.get("tutorial_info", {})

    parts.append(f"**{tutorial.get('title', 'Sequence Example')}**")
    parts.append(f"Type: {tutorial.get('sequence_type', 'Unknown')}")
    parts.append(f"Complexity: {tutorial.get('complexity', 'Unknown')}")

    if tutorial.get("summary"):
        parts.append(f"Summary: {tutorial['summary']}")

    if tutorial.get("key_techniques"):
        parts.append(f"Techniques: {', '.join(tutorial['key_techniques'])}")

    return "\n".join(parts) + "\n"


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


async def validate_code_block(
    ctx: RunContext[PulsePalDependencies], code: str, language: str = "matlab"
) -> str:
    """
    Validate a code block for correct Pulseq function usage.
    Checks function names, capitalization, and namespaces.

    Args:
        code: Code block to validate
        language: Programming language ('matlab' or 'python')

    Returns:
        Validation summary with errors and suggested fixes
    """
    from .code_validator import PulseqCodeValidator

    validator = PulseqCodeValidator()
    result = validator.validate_code(code, language)

    if result.is_valid:
        return "✅ Code validation passed - all Pulseq functions are correct"

    # Format response with errors and fixes
    response = ["❌ Code validation found issues:\n"]

    # Show errors with fixes
    for error in result.errors:
        response.append(f"**Line {error['line']}:** `{error['function']}`")
        response.append(f"  Error: {error['error']}")
        if error.get("suggestion"):
            response.append(f"  Fix: {error['suggestion']}")
        response.append("")

    # If we have fixed code, show it
    if result.fixed_code:
        response.append("\n**Corrected code:**")
        response.append("```" + language)
        response.append(result.fixed_code)
        response.append("```")

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
