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
    
    # Get detected functions from semantic router if available
    detected_functions = None
    if hasattr(ctx.deps, "detected_functions") and ctx.deps.detected_functions:
        detected_functions = ctx.deps.detected_functions
        logger.info(f"Using detected functions: {[f['name'] for f in detected_functions]}")

    # Use new source-aware search with detected functions
    results = await ctx.deps.rag_v2.search_with_source_awareness(
        query=query, 
        sources=sources, 
        forced=forced, 
        source_hints=source_hints,
        detected_functions=detected_functions
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
    """Format code example result for display - provides data, not presentation."""
    parts = []

    def extract_code_from_content(content: str) -> str:
        """Extract code part from content that may have summary---code format."""
        if "---" in content:
            # Split on --- and take the second part (the actual code)
            code_parts = content.split("---", 1)
            if len(code_parts) > 1:
                return code_parts[1].strip()
        return content

    if result.get("source_type") == "DOCUMENTATION_MULTI_PART":
        doc = result.get("document", {})
        parts.append(f"Document: {doc.get('title', 'Document')}")
        parts.append(f"Parts: {doc.get('total_parts', 1)}")
        parts.append(f"Source: {doc.get('url', 'N/A')}")

        # Extract code part if it has the summary---code format
        content = doc.get("content", "")
        content = extract_code_from_content(content)

        # Provide content info and raw content
        if len(content) > 10000:
            parts.append(
                f"Content length: {len(content)} characters (truncated to 10000)"
            )
            content = content[:10000] + "\n[...truncated]"
        else:
            parts.append(f"Content length: {len(content)} characters")

        parts.append("---CONTENT_START---")
        parts.append(content)
        parts.append("---CONTENT_END---")
    else:
        content_info = result.get("content", {})
        parts.append(f"Example: {content_info.get('title', 'Code Example')}")

        if content_info.get("url"):
            parts.append(f"Source: {content_info['url']}")

        # Extract code part if it has the summary---code format
        text = content_info.get("text", "")
        text = extract_code_from_content(text)

        # Check if this looks like code
        is_code = any(
            pattern in text for pattern in ["function", "def ", "class ", "%", "//"]
        )

        # Provide metadata about the content
        content_type = "code" if is_code else "documentation"
        parts.append(f"Content type: {content_type}")

        # Handle truncation for very long content
        if len(text) > 10000:
            parts.append(f"Content length: {len(text)} characters (truncated to 10000)")
            text = text[:10000] + "\n[...truncated]"
        else:
            parts.append(f"Content length: {len(text)} characters")

        parts.append("---CONTENT_START---")
        parts.append(text)
        parts.append("---CONTENT_END---")

    return "\n".join(parts) + "\n"


def format_tutorial_result(result: Dict) -> str:
    """Format tutorial/sequence result for display - provides data, not presentation."""
    parts = []
    tutorial = result.get("tutorial_info", {})
    implementation = result.get("implementation", {})

    # Provide metadata about the sequence
    parts.append(f"Sequence: {tutorial.get('title', 'Sequence Example')}")
    parts.append(f"Type: {tutorial.get('sequence_type', 'Unknown')}")
    parts.append(f"Complexity: {tutorial.get('complexity', 'Unknown')}")

    if tutorial.get("summary"):
        parts.append(f"Summary: {tutorial['summary']}")

    if tutorial.get("key_techniques"):
        parts.append(f"Techniques: {', '.join(tutorial['key_techniques'])}")

    # Provide the code content without prescriptive formatting
    if implementation.get("full_code"):
        full_content = implementation["full_code"]

        # Parse out just the code part after the --- separator
        if "---" in full_content:
            # Split on --- and take the second part (the actual code)
            code_parts = full_content.split("---", 1)
            if len(code_parts) > 1:
                code_only = code_parts[1].strip()
            else:
                code_only = full_content
        else:
            code_only = full_content

        # Count lines for context
        line_count = len(code_only.splitlines())
        parts.append(f"\nCode available: {line_count} lines, MATLAB")
        parts.append("---CODE_START---")
        parts.append(code_only)
        parts.append("---CODE_END---")

    # Include usage guide if available
    usage_guide = result.get("usage_guide", {})
    if usage_guide.get("when_to_use"):
        parts.append(f"Usage context: {usage_guide['when_to_use']}")

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


