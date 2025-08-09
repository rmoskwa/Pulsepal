"""
Simplified tools for Pulsepal agent with intelligent search integration.

Provides a unified tool that intelligently decides when to search the knowledge base
and seamlessly integrates results into responses.

ENHANCED VERSION: Better detection of sequence names and code requests.
Now includes advanced debugging capabilities for Pulseq code.
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
        description="Type of search: 'documentation', 'code', 'sources', or 'auto'",
    )
    match_count: int = Field(5, description="Number of results to return", ge=1, le=20)
    force_search: bool = Field(
        default=False, description="Force search even for general knowledge queries"
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
        "epi": "echo planar imaging EPI sequence implementation",
        "spin echo": "spin echo sequence implementation",
        "gradient echo": "gradient echo GRE sequence implementation",
        "flash": "FLASH gradient echo sequence",
        "tse": "turbo spin echo TSE sequence",
        "fse": "fast spin echo FSE sequence",
        "mprage": "MPRAGE sequence implementation",
        "diffusion": "diffusion weighted imaging sequence",
        # UTE support
        "ute": "UTE ultra short echo time sequence implementation writeUTE",
        "ultrashort": "UTE ultra short echo time sequence implementation",
        "zero te": "zero TE ZTE sequence implementation",
        "zute": "ZUTE zero ultra short echo time sequence",
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
    if (
        any(term in query_lower for term in ["script", "demo", "code"])
        and "implementation" not in query_lower
    ):
        return f"{query} implementation"

    return query


async def verify_pulseq_functions(
    ctx: RunContext[PulsePalDependencies], function_names: list[str]
) -> str:
    """
    CRITICAL: Verify Pulseq function names BEFORE using them in code.
    Always call this before generating any Pulseq code to prevent hallucinations.

    Args:
        function_names: List of function names to verify (e.g., ['seq.calculateKspacePP', 'mr.makeSincPulse'])

    Returns:
        Verification results with corrections for any invalid functions
    """
    from .function_index import MATLAB_FUNCTIONS, COMMON_HALLUCINATIONS
    from .function_similarity import FunctionSimilarity

    results = []
    similarity_checker = FunctionSimilarity()

    for full_name in function_names:
        # Extract the function name from full reference
        # Handle multi-level patterns like mr.aux.quat.multiply
        if "." in full_name:
            parts = full_name.split(".")
            if (
                len(parts) == 4
                and parts[0] == "mr"
                and parts[1] == "aux"
                and parts[2] == "quat"
            ):
                # mr.aux.quat.function pattern
                prefix = ".".join(parts[:3])  # 'mr.aux.quat'
                func_name = parts[3]
            elif len(parts) == 3 and parts[0] == "mr" and parts[1] == "aux":
                # mr.aux.function pattern
                prefix = ".".join(parts[:2])  # 'mr.aux'
                func_name = parts[2]
            else:
                # Standard pattern (mr.*, seq.*, eve.*, tra.*, opt.*)
                prefix, func_name = full_name.rsplit(".", 1)
        else:
            prefix = None
            func_name = full_name

        # Check if it's valid
        is_valid = False
        correct_form = None

        # Check direct calls
        if func_name in MATLAB_FUNCTIONS["direct_calls"]:
            is_valid = True
            correct_form = f"mr.{func_name}" if prefix is None else full_name

        # Check class methods
        if not is_valid and prefix == "seq":
            if func_name in MATLAB_FUNCTIONS["class_methods"].get("Sequence", set()):
                is_valid = True
                correct_form = full_name

        # Check eve.* functions
        if not is_valid and prefix == "eve":
            if func_name in MATLAB_FUNCTIONS.get("eve_functions", set()):
                is_valid = True
                correct_form = full_name

        # Check tra.* functions
        if not is_valid and prefix == "tra":
            if func_name in MATLAB_FUNCTIONS.get("tra_functions", set()):
                is_valid = True
                correct_form = full_name

        # Check mr.aux.quat.* functions
        if not is_valid and prefix == "mr.aux.quat":
            if func_name in MATLAB_FUNCTIONS.get("mr_aux_quat_functions", set()):
                is_valid = True
                correct_form = full_name

        # Check mr.aux.* functions
        if not is_valid and prefix == "mr.aux":
            if func_name in MATLAB_FUNCTIONS.get("mr_aux_functions", set()):
                is_valid = True
                correct_form = full_name

        # Check common hallucinations
        if not is_valid and func_name in COMMON_HALLUCINATIONS:
            correction = COMMON_HALLUCINATIONS[func_name]
            if correction:
                correct_form = f"{prefix}.{correction}" if prefix else correction
            else:
                correct_form = None  # Function doesn't exist

        # Try similarity matching if still not found
        if not is_valid and not correct_form:
            similar = similarity_checker.find_similar(func_name, "")
            if similar:
                correct_form = f"{prefix}.{similar[0][0]}" if prefix else similar[0][0]

        # Build result
        if is_valid:
            results.append(f"✓ {full_name} - VALID")
        elif correct_form:
            results.append(f"✗ {full_name} - INVALID. Use: {correct_form}")
        else:
            results.append(f"✗ {full_name} - DOES NOT EXIST in Pulseq")
            if "gamma" in func_name.lower():
                results.append(
                    "  Note: Define gamma manually as: gamma = 42.5764e6; % Hz/T for 1H"
                )

    return "\n".join(results)


async def search_pulseq_functions_fast(
    ctx: RunContext[PulsePalDependencies], query: str, limit: int = 10
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
            return (
                f"No functions found matching '{query}'. Try a different search term."
            )

        # Format lightweight results for Level 2 response
        output = f"## Function Search Results for: '{query}'\n\n"
        for func in results[:5]:  # Show top 5
            output += f"### {func['name']}\n"
            output += f"**Signature**: `{func['signature']}`\n"
            output += f"**Description**: {func['description']}\n"
            if func.get("is_class_method"):
                output += f"**Usage**: `seq.{func['name']}(...)`\n"
            else:
                output += f"**Usage**: `{func.get('calling_pattern', func['name'] + '(...)')}`\n"
            output += "\n"

        if len(results) > 5:
            output += f"\n*{len(results) - 5} more results available. Use get_function_details for complete parameters.*\n"

        return output

    except Exception as e:
        logger.error(f"Fast function search failed: {e}")
        return "Function search temporarily unavailable. Try again or use search_pulseq_functions."


async def get_function_details(
    ctx: RunContext[PulsePalDependencies], function_names: str | list[str]
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
            if func.get("parameters"):
                output += f"**Parameters**:\n{func['parameters']}\n"
            if func.get("usage_examples"):
                output += f"**Examples**:\n```matlab\n{func['usage_examples']}\n```\n"
            if func.get("returns"):
                output += f"**Returns**: {func['returns']}\n"
            output += "\n---\n"

        return output

    except Exception as e:
        logger.error(f"Function details fetch failed: {e}")
        return "Could not retrieve function details. Use search_pulseq_functions for basic info."


async def get_official_sequence_example(
    ctx: RunContext[PulsePalDependencies],
    sequence_type: str,
    specific_file: str = None,
    exclude_files: list[str] = None,
) -> str:
    """
    Get official Pulseq sequence implementation when users ask "do you have X sequence?" or request sequence code.

    USE THIS TOOL WHEN:
    - User asks "do you have [sequence] sequences?"
    - User requests "[sequence] example" or "[sequence] code"
    - User wants to see any standard MRI sequence implementation
    - User asks for "another" or "different" sequence example

    This tool works in two modes:
    1. Discovery mode (specific_file=None): Returns metadata for top 5 matches, letting you choose the best one
    2. Retrieval mode (specific_file provided): Returns full content for that specific file

    Args:
        sequence_type: Type of sequence requested. Common values:
            - "EPI" for echo planar imaging
            - "SpinEcho" for spin echo sequences
            - "GradientEcho" or "GRE" for gradient echo
            - "TSE" for turbo spin echo
            - "MPRAGE" for MPRAGE sequences
            - "UTE" for ultra-short TE
            - "Spectroscopy" or "PRESS" for spectroscopy sequences
            - "TrueFISP" for balanced SSFP
            - "Spiral" for spiral trajectories
        specific_file: Optional filename to retrieve specific sequence (e.g., "writeEpi.m")
        exclude_files: Optional list of filenames to exclude (use when showing alternatives)

    Returns:
        In discovery mode: Metadata for top 5 matches to help selection
        In retrieval mode: Formatted response with GitHub link, preview, and engagement options

    IMPORTANT: When user asks for "another" example of the same sequence type:
    1. Use the same sequence_type as before
    2. Pass previously shown files in exclude_files
    3. Select from remaining alternatives
    """
    try:
        # Import the guard
        from .official_sequence_guard import get_official_sequence_guard

        guard = get_official_sequence_guard()

        rag_service = get_rag_service()

        # Pass conversation context to rag_service for language preference detection
        if hasattr(ctx, "deps") and hasattr(ctx.deps, "conversation_context"):
            rag_service.conversation_context = ctx.deps.conversation_context
            logger.debug(
                f"Passed conversation context to RAG service, language: {ctx.deps.conversation_context.preferred_language}"
            )

        # Use timeout to prevent long-running queries
        import asyncio

        try:
            result = await asyncio.wait_for(
                rag_service.get_official_sequence(sequence_type, specific_file),
                timeout=10.0,  # Increased timeout for complex queries
            )
            logger.info(
                f"Tool got result for {sequence_type} (specific={specific_file}): {bool(result)}, keys: {result.keys() if result else 'None'}"
            )
        except asyncio.TimeoutError:
            logger.error(f"Tool timeout for {sequence_type} after 10s")
            raise
        except Exception as e:
            logger.error(f"Tool error for {sequence_type}: {str(e)}", exc_info=True)
            raise

        # Log the search event AFTER getting results
        if ctx.deps and hasattr(ctx.deps, "conversation_context"):
            conversation_logger.log_search_event(
                ctx.deps.conversation_context.session_id,
                "official_sequence",
                sequence_type,
                1 if result else 0,
                {"tool": "get_official_sequence_example", "found": bool(result)},
            )

        # Check if we got metadata (discovery mode) or full content (retrieval mode)
        if result and "matches" in result:
            # Discovery mode: Got metadata for top 5 matches
            matches = result["matches"]
            top_sim = result["top_similarity"]

            # Define minimum similarity threshold for relevance
            MIN_SIMILARITY_THRESHOLD = 0.7  # 70% similarity required

            # Check if the best match meets minimum threshold
            if not matches or top_sim < MIN_SIMILARITY_THRESHOLD:
                logger.info(
                    f"Best match similarity {top_sim:.3f} is below threshold {MIN_SIMILARITY_THRESHOLD}"
                )
                return (
                    f"No {sequence_type} sequence examples found in the official Pulseq repository.\n\n"
                    + "The search didn't find any sequences closely matching your request.\n\n"
                    + "Would you like me to:\n"
                    + "1. Search for similar sequence types?\n"
                    + "2. Explain how to implement this sequence from scratch?\n"
                    + "3. Search the broader knowledge base for community examples?"
                )

            # Check if we need to filter by sequence type
            # Map common sequence names to their database types
            sequence_type_map = {
                "spectroscopy": "Spectroscopy",
                "press": "Spectroscopy",
                "mrs": "Spectroscopy",
                "gradientecho": "GradientEcho",
                "gre": "GradientEcho",
                "epi": "EPI",
                "spinecho": "SpinEcho",
                "tse": "TSE",
                "mprage": "MPRAGE",
                "diffusion": "Diffusion",
                "trufisp": "TrueFISP",
                "spiral": "Spiral",
                "ute": "UTE",
                "zte": "ZTE",
                "propeller": "PROPELLER",
                "blade": "PROPELLER",  # BLADE is another name for PROPELLER
            }

            expected_type = sequence_type_map.get(sequence_type.lower(), "")

            # If we have an expected type and the top match doesn't match, filter
            if expected_type and matches:
                top_match_type = matches[0].get("sequence_type", "")
                if top_match_type != expected_type:
                    logger.info(
                        f"Top match type '{top_match_type}' doesn't match expected '{expected_type}', filtering..."
                    )
                    # Filter to only matching types
                    type_filtered = [
                        m
                        for m in matches
                        if m.get("sequence_type", "") == expected_type
                    ]
                    if type_filtered:
                        matches = type_filtered
                        top_sim = matches[0]["similarity"] if matches else 0
                        logger.info(
                            f"Filtered to {len(matches)} {expected_type} sequences"
                        )
                    else:
                        # No matches of the expected type found
                        logger.info(f"No sequences of type '{expected_type}' found")
                        return (
                            f"No {sequence_type} sequence examples found in the official Pulseq repository.\n\n"
                            + f"Note: The closest matches were {top_match_type} sequences, not {expected_type}.\n\n"
                            + "Would you like me to:\n"
                            + f"1. Show the {top_match_type} sequences instead?\n"
                            + "2. Search for similar sequence types?\n"
                            + "3. Explain how to implement {sequence_type} from scratch?"
                        )

            # Find the best match based on Gemini's intelligent selection criteria:
            # 1. If top match similarity is high (>0.8), prefer simpler within 5% similarity
            # 2. Consider user's language preference
            # 3. Default to simplest (shortest) if within threshold

            selected = None
            threshold = 0.05  # 5% similarity threshold

            # Get user's language preference from context
            prefer_python = False
            if hasattr(ctx, "deps") and hasattr(ctx.deps, "conversation_context"):
                lang = getattr(
                    ctx.deps.conversation_context, "preferred_language", None
                )
                prefer_python = lang == "python"

            # Apply exclusion filter if provided
            if exclude_files:
                exclude_files = (
                    exclude_files
                    if isinstance(exclude_files, list)
                    else [exclude_files]
                )
                # Filter out any files that have been shown before
                matches = [m for m in matches if m["file_name"] not in exclude_files]
                if not matches:
                    logger.warning(
                        f"All matches for {sequence_type} have been excluded"
                    )
                    return f"All available {sequence_type} sequences have already been shown. Try a different sequence type."

            # First, filter by language preference if strong match exists
            lang_filtered = [
                m
                for m in matches
                if m["language"] == ("Python" if prefer_python else "MATLAB")
            ]
            if not lang_filtered:
                lang_filtered = matches  # Use all if no language match

            # Find simplest within threshold
            for match in lang_filtered:
                if match["similarity"] >= (top_sim - threshold):
                    if (
                        not selected
                        or (
                            match["complexity"] == "basic"
                            and selected["complexity"] == "advanced"
                        )
                        or (
                            match["complexity"] == selected["complexity"]
                            and match["content_length"] < selected["content_length"]
                        )
                    ):
                        selected = match

            # If no selection, just use the top match
            if not selected:
                selected = matches[0]

            logger.info(
                f"Selected {selected['file_name']} from {len(matches)} matches (sim={selected['similarity']:.3f}, complexity={selected['complexity']})"
            )

            # Store information about alternatives for the response
            # Filter to only show alternatives of the same sequence type
            selected_type = selected.get("sequence_type", "")
            if selected_type:
                # Only show alternatives with matching sequence type
                other_matches = [
                    m
                    for m in matches
                    if m["file_name"] != selected["file_name"]
                    and m.get("sequence_type", "") == selected_type
                ]
            else:
                # Fallback if no sequence_type field
                other_matches = [
                    m for m in matches if m["file_name"] != selected["file_name"]
                ]

            alternatives_info = {
                "selected": selected["file_name"],
                "selected_type": selected_type,
                "alternatives": other_matches[:3]
                if other_matches
                else [],  # Top 3 alternatives of same type
            }

            # Now retrieve the full content for the selected file
            result = await asyncio.wait_for(
                rag_service.get_official_sequence(sequence_type, selected["file_name"]),
                timeout=10.0,
            )

            if not result or "content" not in result:
                logger.warning(
                    f"Failed to retrieve content for selected file: {selected['file_name']}"
                )
                result = None

        if not result or "content" not in result:
            # No official example - automatically search for community examples
            logger.info(
                f"No official example for '{sequence_type}', searching community examples..."
            )

            # Try different query variations for better fallback
            fallback_queries = [
                f"{sequence_type} sequence implementation",
                f"{sequence_type} MRI sequence example",
                f"{sequence_type} Pulseq code",
            ]

            for fallback_query in fallback_queries:
                fallback_result = await search_pulseq_knowledge(
                    ctx, fallback_query, search_type="code", match_count=3
                )

                # Check if we got meaningful results
                if fallback_result and "No code examples found" not in fallback_result:
                    return (
                        f"## {sequence_type} Sequence (Community Examples)\n\n"
                        + "*Note: No official example found. Showing community implementations:*\n\n"
                        + fallback_result
                    )

            # If all fallbacks fail
            return (
                f"No {sequence_type} sequence examples found in the database.\n\n"
                + "Would you like me to:\n"
                + "1. Explain how to implement this sequence from scratch?\n"
                + "2. Search for similar sequence types?\n"
                + "3. Provide the general structure and key functions needed?"
            )

        # Check if user is requesting full code from official sequence
        # Get the current user query from context
        user_query = ""
        if hasattr(ctx, "deps") and hasattr(ctx.deps, "conversation_context"):
            # Use conversation_history instead of messages
            history = ctx.deps.conversation_context.conversation_history
            if history:
                # Find the last user message
                for entry in reversed(history):
                    if entry.get("role") == "user":
                        user_query = entry.get("content", "")
                        break

        # Log for tracking purposes only (no longer preventing display)
        source_info = {
            "sequence_type": sequence_type,
            "file_name": result.get("file_name", ""),
            "url": result.get("url", ""),
            "ai_summary": result.get("ai_summary", ""),
            "table_name": "official_sequence_examples",
            "source": "github.com/pulseq/pulseq",
        }

        # Check if this would have triggered protection (for logging only)
        needs_protection, _ = guard.check_and_guard(user_query, source_info)

        if needs_protection:
            # Just log it, don't prevent display
            logger.info(
                f"Official sequence request logged for {sequence_type} (protection disabled)"
            )

        # Parse content if it contains the summary separator
        content = result.get("content", "")
        if not content:
            # If content is missing, it means we only got the summary
            # This happens when the tool fetches the wrong fields
            logger.warning(
                f"No content in result for {sequence_type}, result keys: {result.keys()}"
            )
            # Try to use ai_summary if available
            if "ai_summary" in result and result["ai_summary"]:
                return (
                    f"## {sequence_type} Sequence Summary\n\n"
                    f"{result['ai_summary']}\n\n"
                    f"*Note: Full implementation code not available. "
                    f"Try search_pulseq_knowledge for complete examples.*"
                )
            return (
                f"Found {sequence_type} but content is missing.\n"
                f"Try search_pulseq_knowledge for community examples."
            )
        if "---" in content:
            # Extract just the code part after the summary
            parts = content.split("---", 1)
            if len(parts) > 1:
                content = parts[1].strip()

        # Provide full code when appropriate
        lines = content.split("\n")

        # Build response with source reference
        file_name = result.get("file_name", "Unknown")
        # Extract just the filename without path
        short_name = file_name.split("/")[-1] if "/" in file_name else file_name

        output = f"## {short_name}\n\n"
        output += f"🔗 **GitHub**: https://github.com/pulseq/pulseq/blob/master/{file_name}\n\n"

        # Add the AI summary if available for context
        if "ai_summary" in result and result["ai_summary"]:
            summary = result["ai_summary"]
            # Take first 2-3 sentences of summary
            summary_sentences = summary.split(". ")[:3]
            brief_summary = ". ".join(summary_sentences)
            if not brief_summary.endswith("."):
                brief_summary += "."
            output += f"**Overview**: {brief_summary}\n\n"

        # Show the complete code
        output += "### Complete Code:\n"
        output += "```matlab\n"
        output += content
        output += "\n```\n\n"

        # Add engagement questions based on sequence type
        output += "### How Can I Help You With This Sequence?\n\n"
        output += "Would you like me to:\n"
        output += f"1. **Explain** the key concepts and physics behind this {sequence_type} sequence?\n"
        output += "2. **Modify** specific parameters for your application (FOV, resolution, TR/TE)?\n"
        output += "3. **Create** a custom version with your requirements?\n"
        output += "4. **Debug** issues if you're implementing this sequence?\n"
        output += f"5. **Compare** this with other {sequence_type} variants?\n\n"

        # Add information about alternatives if available
        if "alternatives_info" in locals() and alternatives_info.get("alternatives"):
            output += f"📚 **Other {sequence_type} examples available:**\n"
            for alt in alternatives_info["alternatives"][
                :3
            ]:  # Show up to 3 alternatives
                complexity_note = " (simpler)" if alt["complexity"] == "basic" else ""
                output += f"- `{alt['file_name']}`{complexity_note}\n"
            output += "\n*To see another example, just ask!*\n\n"

        output += "💡 *Tip: You can view and download the complete code from the GitHub link above.*"

        return output

    except asyncio.TimeoutError:
        logger.warning(
            f"Official sequence fetch timed out for {sequence_type} (10s timeout)"
        )
        # Fallback to search when timeout occurs
        logger.info(f"Falling back to search_pulseq_knowledge for {sequence_type}")
        return await search_pulseq_knowledge(
            ctx, f"{sequence_type} sequence example", search_type="code"
        )
    except Exception as e:
        logger.error(f"Official sequence fetch failed: {e}")
        # Fallback to broader search
        return await search_pulseq_knowledge(
            ctx, f"{sequence_type} sequence example", search_type="code"
        )


# Removed timeout temporarily to debug
async def search_pulseq_knowledge(
    ctx: RunContext[PulsePalDependencies],
    query: str,
    search_type: str = "auto",
    match_count: int = 5,
    force_search: bool = False,
) -> str:
    """
    Search the broader Pulseq knowledge base for documentation, functions, or community examples.

    USE THIS TOOL ONLY WHEN:
    - get_official_sequence_example() doesn't have the sequence you need
    - User asks for specific function documentation (mr.makeSincPulse, etc.)
    - User needs conceptual explanations or tutorials
    - Searching for non-standard or custom implementations

    DO NOT USE THIS for standard sequences like EPI, GRE, TSE, etc. - use get_official_sequence_example instead.

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
            force_search=force_search,
        )

        # Get RAG service
        rag_service = get_rag_service()

        # Use enhanced perform_rag_query for auto routing
        if params.search_type == "auto":
            # Check if async method exists, otherwise use sync version
            if hasattr(
                rag_service, "perform_rag_query"
            ) and asyncio.iscoroutinefunction(rag_service.perform_rag_query):
                results = await rag_service.perform_rag_query(
                    query=params.query,
                    search_type="auto",
                    match_count=params.match_count,
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
                        use_hybrid=True,
                    ),
                )

            # Log search event for debugging
            if ctx.deps and hasattr(ctx.deps, "conversation_context"):
                conversation_logger.log_search_event(
                    ctx.deps.conversation_context.session_id,
                    "enhanced_auto",
                    params.query,
                    1 if results else 0,
                    {"search_type": "auto", "enhanced": True},
                )

            return results

        # Determine search strategy based on type
        if params.search_type == "auto":
            # Intelligent routing based on query content
            query_lower = query.lower()

            # ENHANCED: Common MRI sequence names - prioritize code search
            sequence_indicators = [
                "epi",
                "echo planar",
                "spin echo",
                "gradient echo",
                "gre",
                "flash",
                "tse",
                "turbo spin",
                "fse",
                "fast spin",
                "mprage",
                "diffusion",
                "dwi",
                "dti",
                "bold",
                "fmri",
                "spiral",
                "radial",
                "propeller",
                "blade",
                "fisp",
                "ssfp",
                "ir",
                "inversion recovery",
                "stir",
                "flair",
                "tof",
                "pc",
                # UTE variants
                "ute",
                "ultra short",
                "ultrashort",
                "zero te",
                "zte",
                "zute",
            ]

            # Code/script request indicators
            code_request_indicators = [
                "script",
                "demo",
                "code",
                "example",
                "implement",
                "sample",
                "show me",
                "give me",
                "provide",
                "write",
                "create",
            ]

            # Pulseq-specific function indicators
            function_indicators = [
                "mr.",
                "makeblock",
                "makegauss",
                "makearb",
                "addblock",
                "calcgradient",
                "calcduration",
                "sequence.",
                "pypulseq",
                "makesinc",
                "maketrapezoid",
                "makeadc",
                ".seq",
            ]

            # Check if asking for sequence example/script
            has_sequence = any(seq in query_lower for seq in sequence_indicators)
            has_code_request = any(
                req in query_lower for req in code_request_indicators
            )
            has_function = any(func in query_lower for func in function_indicators)

            # Prioritize code search for sequence requests
            if (
                (has_sequence and has_code_request)
                or has_function
                or (
                    "pulseq" in query_lower
                    and any(
                        term in query_lower for term in ["example", "script", "demo"]
                    )
                )
            ):
                search_type = "code"
                # Enhance the query for better results
                query = enhance_sequence_query(params.query)
                logger.info(f"Enhanced query from '{params.query}' to '{query}'")

            # Documentation indicators (only if not already code)
            elif any(
                indicator in query_lower
                for indicator in [
                    "tutorial",
                    "guide",
                    "documentation",
                    "manual",
                    "reference",
                ]
            ):
                search_type = "documentation"

            # Source discovery
            elif any(
                indicator in query_lower
                for indicator in [
                    "available",
                    "sources",
                    "repositories",
                    "what sources",
                    "list",
                ]
            ):
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
            if ctx.deps and hasattr(ctx.deps, "conversation_context"):
                conversation_logger.log_search_event(
                    ctx.deps.conversation_context.session_id,
                    "sources",
                    query,
                    results.count("**") if results else 0,
                    {"search_type": search_type},
                )

        elif search_type == "code":
            # First check for official sequences if it's a sequence request
            if any(
                seq in query.lower()
                for seq in [
                    "epi",
                    "spin echo",
                    "gradient echo",
                    "tse",
                    "mprage",
                    "ute",
                    "haste",
                    "trufi",
                    "press",
                    "spiral",
                ]
            ):
                # Use async version of search_code_examples if available
                if hasattr(
                    rag_service, "search_code_examples"
                ) and asyncio.iscoroutinefunction(rag_service.search_code_examples):
                    results = await rag_service.search_code_examples(
                        query=query, match_count=params.match_count
                    )
                else:
                    loop = asyncio.get_event_loop()
                    results = await loop.run_in_executor(
                        None,
                        lambda: rag_service.search_code_examples(
                            query=query, match_count=params.match_count
                        ),
                    )
                # Log and return early - we already have formatted results
                logger.info(f"Code search completed for sequence: {query}")
                if ctx.deps and hasattr(ctx.deps, "conversation_context"):
                    conversation_logger.log_search_event(
                        ctx.deps.conversation_context.session_id,
                        "code",
                        query,
                        1 if results and "No code examples found" not in results else 0,
                        {"search_type": search_type, "sequence_search": True},
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
                        keyword_query_override=seq_type,
                    )

                    # Apply the new scoring method
                    if raw_results:
                        scored_results = []
                        for result in raw_results:
                            # Use the rag_service's scoring method
                            score = rag_service._score_sequence_relevance(
                                result, seq_type
                            )

                            if score > 0 or len(scored_results) < 10:
                                result["relevance_score"] = score
                                scored_results.append(result)

                        scored_results.sort(
                            key=lambda x: x.get("relevance_score", 0), reverse=True
                        )
                        raw_results = scored_results
                else:
                    # Hybrid search with optional filtering
                    filtered_query = (
                        metadata.get("filtered_query")
                        if strategy == "hybrid_filtered"
                        else None
                    )
                    raw_results = rag_service.supabase_client.perform_hybrid_search(
                        query=query,
                        match_count=50,  # Reduced for better performance
                        search_type="code_examples",
                        keyword_query_override=filtered_query,
                    )

            # Format results - always shows top result directly
            formatted_results, _ = rag_service.format_code_results_interactive(
                raw_results, query
            )
            results = formatted_results
            logger.info(f"Code search completed for: {query}")

            # Log search event for debugging
            if ctx.deps and hasattr(ctx.deps, "conversation_context"):
                results_count = len(raw_results) if raw_results else 0
                conversation_logger.log_search_event(
                    ctx.deps.conversation_context.session_id,
                    "code",
                    query,
                    results_count,
                    {
                        "search_type": search_type,
                        "enhanced_query": query != params.query,
                    },
                )

            # If no results and we enhanced the query, be transparent about trying alternatives
            if "No code examples found" in results and params.query != query:
                # Be transparent about the search enhancement
                transparency_msg = f"\n*Note: No exact matches for '{params.query}'. "
                transparency_msg += f"I searched for related term: '{query}'*\n\n"

                # Try original query if we enhanced it
                # Use the same smart strategy for the fallback
                fallback_strategy, fallback_metadata = (
                    rag_service.classify_search_strategy(params.query)
                )
                if fallback_strategy == "vector_enhanced":
                    # Same logic as above for enhanced search
                    seq_type = fallback_metadata.get("sequence_type", "")
                    fallback_raw_results = (
                        rag_service.supabase_client.perform_hybrid_search(
                            query=params.query,
                            match_count=50,  # Reduced for better performance
                            search_type="code_examples",
                            keyword_query_override=seq_type,
                        )
                    )
                    # Apply the new scoring method
                    if fallback_raw_results:
                        scored = []
                        for r in fallback_raw_results:
                            score = rag_service._score_sequence_relevance(r, seq_type)
                            if score > 0 or len(scored) < 10:
                                r["relevance_score"] = score
                                scored.append(r)
                        scored.sort(
                            key=lambda x: x.get("relevance_score", 0), reverse=True
                        )
                        fallback_raw_results = scored
                else:
                    fallback_filtered = (
                        fallback_metadata.get("filtered_query")
                        if fallback_strategy == "hybrid_filtered"
                        else None
                    )
                    fallback_raw_results = (
                        rag_service.supabase_client.perform_hybrid_search(
                            query=params.query,
                            match_count=50,  # Reduced for better performance
                            search_type="code_examples",
                            keyword_query_override=fallback_filtered,
                        )
                    )
                logger.info(f"Fallback search with original query: {params.query}")

                if fallback_raw_results:
                    fallback_formatted, _ = rag_service.format_code_results_interactive(
                        fallback_raw_results, params.query
                    )
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
                query=params.query, match_count=params.match_count, use_hybrid=True
            )
            logger.info(f"Documentation search completed for: {params.query}")

            # Log search event for debugging
            if ctx.deps and hasattr(ctx.deps, "conversation_context"):
                results_count = (
                    0 if "No documentation found" in results else results.count("###")
                )
                conversation_logger.log_search_event(
                    ctx.deps.conversation_context.session_id,
                    "documentation",
                    params.query,
                    results_count,
                    {"search_type": search_type},
                )

        return results

    except Exception as e:
        logger.error(f"Pulseq knowledge search failed: {e}")

        # Graceful fallback to web search
        try:
            web_service = get_web_search_service()

            if search_type == "code":
                web_results = web_service.search_pulseq_resources(
                    query, resource_type="example"
                )
            else:
                web_results = web_service.search_mri_information(
                    query, max_results=match_count
                )

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
    match_count: int = 5,
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
            if language not in ["matlab", "python", "cpp"]:
                language = None

        # Get RAG service
        rag_service = get_rag_service()

        # Search API functions
        results = rag_service.search_api_functions(
            query=query, match_count=match_count, language_filter=language
        )

        logger.info(f"API function search completed for: {query}")

        # Log search event for debugging
        if ctx.deps and hasattr(ctx.deps, "conversation_context"):
            results_count = (
                0 if "No API functions found" in results else results.count("###")
            )
            conversation_logger.log_search_event(
                ctx.deps.conversation_context.session_id,
                "api_functions",
                query,
                results_count,
                {"language_filter": language, "match_count": match_count},
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
                use_hybrid=True,
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
    ctx: RunContext[PulsePalDependencies], query: str, match_count: int = 10
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
        results = rag_service.search_all_sources(query=query, match_count=match_count)

        logger.info(f"Unified search completed for: {query}")

        # Log search event for debugging
        if ctx.deps and hasattr(ctx.deps, "conversation_context"):
            # Count total results across all sources
            results_count = (
                results.count("###")
                + results.count("🔧")
                + results.count("💻")
                + results.count("📚")
            )
            conversation_logger.log_search_event(
                ctx.deps.conversation_context.session_id,
                "unified",
                query,
                results_count,
                {"match_count": match_count},
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
    match_count: int = 5,
) -> str:
    """
    Legacy tool: Use search_pulseq_knowledge instead.
    Search the RAG database for Pulseq documentation and information.
    """
    logger.warning(
        "perform_rag_query is deprecated. Use search_pulseq_knowledge instead."
    )
    return await search_pulseq_knowledge(
        ctx, query, search_type="documentation", match_count=match_count
    )


async def search_code_examples(
    ctx: RunContext[PulsePalDependencies],
    query: str,
    source_id: Optional[str] = None,
    match_count: int = 5,
) -> str:
    """
    DEPRECATED: Use search_pulseq_knowledge with search_type='code' instead.
    Search for Pulseq code examples and implementations.
    NOTE: This searches the code_examples_legacy table which is deprecated.
    """
    logger.warning(
        "search_code_examples is deprecated. Use search_pulseq_knowledge instead."
    )
    return await search_pulseq_knowledge(
        ctx, query, search_type="code", match_count=match_count
    )


async def get_available_sources(ctx: RunContext[PulsePalDependencies]) -> str:
    """
    Legacy tool: Use search_pulseq_knowledge instead.
    Get list of available documentation sources in the RAG database.
    """
    logger.warning(
        "get_available_sources is deprecated. Use search_pulseq_knowledge instead."
    )
    return await search_pulseq_knowledge(
        ctx, "available sources", search_type="sources"
    )


# ============================================================================
# NEW DEBUGGING TOOLS
# ============================================================================


async def trace_physics_to_code(
    ctx: RunContext[PulsePalDependencies],
    physics_problem: str,
    code_snippet: Optional[str] = None,
) -> str:
    """
    Category 2: Trace from MRI physics problem to responsible code.
    Uses concept_mapper for hints on common patterns, but handles novel problems too.

    Args:
        physics_problem: Description like "k-space trajectory wrong" or any novel issue
        code_snippet: Optional relevant code section

    Returns:
        Physics analysis with code investigation guidance
    """
    from .concept_mapper import ConceptMapper

    mapper = ConceptMapper()
    mapping = mapper.map_problem_to_code(physics_problem)

    # Check if this is a novel problem
    if mapping.get("is_novel", False):
        # This is a novel problem - provide systematic framework
        output = [
            f"## Novel Problem Analysis: {physics_problem}\n",
            "This issue isn't in my common patterns database, but I can analyze it systematically:\n",
            "### Systematic Debugging Approach:",
            "1. **Physics Analysis**: What fundamental MRI principles are involved?",
            "2. **Sequence Components**: Which parts of the sequence affect this?",
            "3. **Pulseq Implementation**: What functions control these components?",
            "4. **Validation**: How to verify correct implementation?\n",
            "### General Investigation Areas:",
        ]

        for fix in mapping.get("common_fixes", []):
            output.append(f"  - {fix}")

        output.append("\n### Next Steps:")
        output.append(
            "Please share your code and I'll apply physics reasoning to identify the issue."
        )

    else:
        # We have hints from the mapper - use them
        output = [
            f"## Physics Analysis: {physics_problem}\n",
            f"**Physics Principle**: {mapping['physics']}\n",
            f"**Responsible Elements**: {', '.join(mapping['sequence_elements'])}\n",
            f"**Check These Functions**: {', '.join(mapping['pulseq_functions'])}\n",
            "**Common Issues**:",
        ]

        for issue in mapping.get("common_issues", []):
            output.append(f"  - {issue}")

    # If code provided, analyze it regardless of whether pattern exists
    if code_snippet:
        output.append("\n## Analyzing Your Code:")
        from .debug_analyzer import PulseqDebugAnalyzer

        analyzer = PulseqDebugAnalyzer()

        # This will use physics reasoning, not just patterns
        results = analyzer.debug_concept(physics_problem, code_snippet)

        if results["issues"]:
            for issue in results["issues"]:
                output.append(f"- **Issue**: {issue.issue_description}")
                output.append(f"  **Solution**: {issue.solution}")

        if results["reasoning"]:
            output.append(f"\n### Physics Reasoning:\n{results['reasoning']}")

    return "\n".join(output)


async def analyze_user_code(
    ctx: RunContext[PulsePalDependencies],
    code: str,
    problem_description: Optional[str] = None,
) -> str:
    """
    Analyze user's code for both syntax and conceptual issues.
    Handles both common patterns and novel problems.

    Args:
        code: The user's Pulseq code to analyze
        problem_description: Optional description (can be novel)

    Returns:
        Comprehensive debugging analysis
    """
    from .debug_analyzer import PulseqDebugAnalyzer

    analyzer = PulseqDebugAnalyzer()
    results = analyzer.analyze_code(code, problem_description)

    output = []

    # Syntax errors (always deterministic)
    if results["syntax_errors"]:
        output.append("## Syntax Issues Found:\n")
        for issue in results["syntax_errors"]:
            output.append(f"Line {issue.line_number}: {issue.explanation}")
            output.append(f"❌ {issue.incorrect_usage}")
            output.append(f"✅ {issue.correct_usage}\n")

    # Conceptual issues (may be novel)
    if results["conceptual_issues"]:
        output.append("## Conceptual Analysis:\n")

        # Check if this was a novel problem
        if any(
            "Systematic analysis" in issue.physics_concept
            for issue in results["conceptual_issues"]
        ):
            output.append("*Note: Analyzing novel problem using physics principles*\n")

        for issue in results["conceptual_issues"]:
            output.append(f"**Physics Concept**: {issue.physics_concept}")
            output.append(f"**Issue**: {issue.issue_description}")
            output.append(f"**Solution**: {issue.solution}\n")

    # Add physics reasoning if available
    if results.get("physics_reasoning"):
        output.append("## Physics Reasoning:")
        output.append(results["physics_reasoning"])

    return "\n".join(output)


async def check_function_syntax(
    ctx: RunContext[PulsePalDependencies], function_call: str
) -> str:
    """
    Quick syntax check for a specific Pulseq function call.

    Args:
        function_call: Function name like 'mr.makeTrapezoid' or 'seq.write'

    Returns:
        Validation result with corrections if needed
    """
    from .syntax_validator import SyntaxValidator

    validator = SyntaxValidator()
    result = validator.validate_function_call(function_call)

    if result["is_valid"]:
        return f"✅ `{function_call}` is valid Pulseq syntax."
    else:
        output = [f"❌ `{function_call}` is invalid."]
        if result.get("correct_form"):
            output.append(f"✅ Correct form: `{result['correct_form']}`")
        output.append(f"Explanation: {result['explanation']}")
        return "\n".join(output)
