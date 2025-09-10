"""
Simplified tools for Pulsepal agent with modern RAG service v2.

Provides tools that use the simplified retrieval-only RAG service,
leaving all intelligence to the LLM.
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from pydantic_ai import RunContext, ModelRetry

from .conversation_logger import get_conversation_logger
from .dependencies import PulsePalDependencies
from .rag_service import ModernPulseqRAG

# Agent will be set by main_agent.py after creation
pulsepal_agent = None

logger = logging.getLogger(__name__)
conversation_logger = get_conversation_logger()

# Cache for table metadata with TTL
_table_metadata_cache: Optional[Tuple[datetime, List[Dict]]] = None
_TABLE_METADATA_TTL = timedelta(hours=1)


async def get_rag_service(ctx: RunContext[PulsePalDependencies]) -> ModernPulseqRAG:
    """
    Get or initialize RAG service with proper error handling.

    This centralizes RAG initialization to avoid code duplication and
    ensures proper error handling.

    Args:
        ctx: PydanticAI run context with dependencies

    Returns:
        Initialized ModernPulseqRAG instance

    Raises:
        RuntimeError: If RAG service initialization fails
    """
    if not hasattr(ctx.deps, "rag_v2"):
        try:
            ctx.deps.rag_v2 = ModernPulseqRAG()
            logger.info("RAG service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            raise RuntimeError(f"RAG service initialization failed: {e}")

    return ctx.deps.rag_v2


def validate_database_identifier(
    identifier: str, identifier_type: str = "column"
) -> bool:
    """
    Validate database identifiers (table/column names) to prevent SQL injection.

    Uses a strict allowlist of characters and patterns that are safe for
    database identifiers.

    Args:
        identifier: The identifier to validate
        identifier_type: Type of identifier ("table" or "column")

    Returns:
        True if identifier is safe, False otherwise
    """
    # Maximum length check
    if len(identifier) > 63:  # PostgreSQL identifier limit
        return False

    # Check for empty or None
    if not identifier:
        return False

    # Strict pattern: only alphanumeric, underscore, and doesn't start with digit
    # This is more restrictive than PostgreSQL allows but safer
    pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"

    if not re.match(pattern, identifier):
        return False

    # Check against SQL keywords that could be dangerous
    dangerous_keywords = {
        "select",
        "insert",
        "update",
        "delete",
        "drop",
        "create",
        "alter",
        "grant",
        "revoke",
        "union",
        "where",
        "from",
        "join",
        "exec",
        "execute",
        "script",
        "javascript",
        "eval",
        "function",
        "procedure",
    }

    if identifier.lower() in dangerous_keywords:
        return False

    return True


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


def _validate_search_relevance(
    results: Dict, table: str, query: str, ctx: Optional[RunContext] = None
) -> None:
    """
    Validate search result relevance and raise ModelRetry if scores are too low.

    This validation works cooperatively with other validators in the system:
    - Only validates RAG search relevance, not function validation or SQL errors
    - Provides specific guidance for improving search quality
    - Allows other validators to handle their specific domains

    Args:
        results: Search results dictionary from RAG service
        table: The table that was searched
        query: The original search query

    Raises:
        ModelRetry: If relevance scores are too low, prompting Gemini to search more broadly
    """
    # Skip validation if results indicate an error already handled elsewhere
    if results.get("error"):
        return  # Let other error handlers deal with this

    # Define minimum acceptable relevance thresholds
    # Based on research and empirical testing:
    # - BM25: Unbounded scores, query-dependent. Our empirical data shows very weak matches < 0.02
    # - Vector: We check count==0 rather than similarity scores (would be <0.3 if available)
    # - BGE Reranker: Raw scores unbounded (can be negative), normalized [0,1] via sigmoid
    #   When normalized, very low relevance < 0.01, very high > 0.99
    MIN_BM25_SCORE = (
        0.02  # Based on GE scanner case: mean 0.0153 indicated poor keyword match
    )
    MIN_RERANK_SCORE = (
        1.0  # For raw BGE scores: negative=irrelevant, 0-1=weak, >1=relevant
    )
    # In our case (raw scores), 2.408 was considered acceptable despite other poor metrics

    # Check if we have search metadata
    metadata = results.get("search_metadata", {})
    if not metadata:
        return  # No metadata to validate

    # Skip validation if this was already a retry from another validation
    if metadata.get("is_retry_search", False):
        return  # Avoid validation loops

    # Also skip if we've already validated this query recently in this context
    if ctx and hasattr(ctx, "_recent_validation_queries"):
        query_key = f"{query[:50]}_{table}"
        if query_key in ctx._recent_validation_queries:
            return  # Already validated this query/table combo

    # Extract performance metrics if available
    performance = metadata.get("performance") or {}

    # Check for specific indicators of poor results
    poor_relevance_indicators = []

    # 1. Check if we got 0 results from vector search
    vector_count = performance.get("vector_count")
    if vector_count is not None and vector_count == 0:
        poor_relevance_indicators.append("No semantic matches found in vector search")

    # 2. Check BM25 scores if available from RRF fusion stats
    rrf_stats = performance.get("rrf_stats")
    if rrf_stats and isinstance(rrf_stats, dict):
        score_mean = rrf_stats.get("score_mean")
        if score_mean is not None and score_mean < MIN_BM25_SCORE:
            poor_relevance_indicators.append(
                f"BM25 keyword scores very low (mean: {score_mean:.4f})"
            )

    # 3. Check reranker scores if available - now from search_metadata
    rerank_stats = metadata.get("rerank_stats")  # Changed from performance to metadata
    if rerank_stats and isinstance(rerank_stats, dict):
        top_score = rerank_stats.get("top_score")
        # BGE reranker scores: negative=irrelevant, 0-1=weak, >1=relevant
        if top_score is not None:
            if top_score < 0:
                # Negative scores are a strong indicator of irrelevance
                poor_relevance_indicators.append(
                    f"Reranker indicates irrelevant results (score: {top_score:.2f})"
                )
            elif top_score < MIN_RERANK_SCORE:
                poor_relevance_indicators.append(
                    f"Reranker confidence low (score: {top_score:.2f})"
                )

    # 4. Check total results - if very few results, might be searching wrong table
    total_results = results.get("total_results", 0)
    if total_results == 0 and table not in ["auto", None]:
        # Zero results is a strong indicator
        poor_relevance_indicators.append(
            f"No results found in table '{table}' (searching wrong table?)"
        )
    elif total_results < 3 and table not in ["auto", None]:
        poor_relevance_indicators.append(
            f"Only {total_results} results found in table '{table}'"
        )

    # Only raise ModelRetry if we have strong evidence of poor search quality
    # Be conservative to avoid conflicting with other validators
    should_retry = False
    retry_reason = None

    # Check all conditions independently (not elif chain)

    # Case 1: Negative reranker score (strong indicator of irrelevance)
    if rerank_stats and isinstance(rerank_stats, dict):
        top_score = rerank_stats.get("top_score")
        if top_score is not None and top_score < 0:
            should_retry = True
            retry_reason = f"negative rerank score: {top_score}"
            logger.info(f"Triggering retry due to {retry_reason}")

    # Case 2: Zero results with specific table (strong signal, check before multiple indicators)
    if not should_retry and total_results == 0 and table not in ["auto", None, ""]:
        # Zero results in a specific table is a strong enough signal on its own
        should_retry = True
        retry_reason = f"zero results in table '{table}'"
        logger.info(f"Triggering retry due to {retry_reason}")

    # Case 3: Multiple indicators of poor relevance (2 or more)
    if not should_retry and len(poor_relevance_indicators) >= 2:
        should_retry = True
        retry_reason = f"{len(poor_relevance_indicators)} poor relevance indicators"
        logger.info(f"Triggering retry due to {retry_reason}")

    if should_retry:
        # Check if it's specifically a negative reranker score case
        if rerank_stats and rerank_stats.get("top_score", 0) < 0:
            retry_message = (
                "⚠️ **Search Results Not Relevant**\n\n"
                "The neural reranker indicates these results are not relevant to your query "
                f"(relevance score: {rerank_stats.get('top_score', 0):.2f}).\n\n"
                "This typically means:\n"
                "• The information uses different terminology\n"
                "• You're searching in the wrong table\n"
                "• The concept may not exist in the knowledge base\n\n"
                "**Current search metrics:**\n"
            )
        else:
            retry_message = (
                "🔍 **Low Relevance Scores Detected**\n\n"
                "The search results have low relevance scores, suggesting the information "
                "might be in a different location.\n\n"
                "**Current search metrics:**\n"
            )

        for indicator in poor_relevance_indicators:
            retry_message += f"  • {indicator}\n"

        retry_message += "\n**Available search options to broaden your search:**\n\n"

        if table not in ["auto", None]:
            retry_message += (
                f"📊 **Currently searching:** `{table}` table only\n\n"
                f"**Try these alternative tables:**\n\n"
                f"  🌐 **`table='auto'`** - Search ALL tables comprehensively\n"
                f"     Best for: General queries, unknown locations\n\n"
                f"  📝 **`table='pulseq_sequences'`** - Complete MRI sequences\n"
                f"     Contains: Full .m files, GRE, EPI, TSE, diffusion sequences\n\n"
                f"  📖 **`table='api_reference'`** - Pulseq function documentation\n"
                f"     Contains: mr.* and seq.* functions, parameters, usage\n\n"
                f"  🧩 **`table='sequence_chunks'`** - Code segments by concept\n"
                f"     Contains: RF pulses, gradients, k-space trajectories\n\n"
                f"  🔧 **`table='crawled_code'`** - Helper functions & utilities\n"
                f"     Contains: Custom functions, calculations, processing\n\n"
                f"  📄 **`table='crawled_docs'`** - Guides & documentation\n"
                f"     Contains: Tutorials, papers, troubleshooting, examples\n\n"
            )
        else:
            # Even in auto mode, if scores are low, suggest different query strategies
            retry_message += (
                "**The search covered all tables but found weak matches.**\n\n"
                "**Consider these approaches:**\n"
                "  🔄 **Rephrase with different terminology**\n"
                "     Example: 'gradient' → 'makeTrapezoid', 'RF pulse' → 'makeBlockPulse'\n\n"
                "  ✨ **Use Pulseq-specific terms**\n"
                "     Example: Instead of 'MRI sequence', try 'seq.write' or 'mr.opts'\n\n"
                "  🎯 **Search for specific functions**\n"
                "     Example: 'mr.makeExtendedTrapezoid' or 'seq.addBlock'\n\n"
                "  📦 **Break complex queries into parts**\n"
                "     Example: Search 'diffusion' first, then 'b-value calculation'\n"
            )

        # Add specific guidance for negative rerank scores
        if rerank_stats and rerank_stats.get("top_score", 0) < 0:
            retry_message += (
                "\n🔬 **Alternative Tools to Try:**\n\n"
                "  🔍 **`lookup_pulseq_function`** - Direct function verification\n"
                "     Use when: You know the exact function name\n\n"
                "  🗺️ **`find_relevant_tables`** - Semantic table discovery\n"
                "     Use when: Unsure which table contains your information\n\n"
                "  📋 **`get_table_schemas`** - Explore table structure\n"
                "     Use when: You need to understand what's available\n"
            )

        if total_results == 0:
            retry_message += (
                "\n💡 **No results found** - The information might:\n"
                "  • Use different terminology than your query\n"
                "  • Be in a different table than expected\n"
                "  • Require a more general search term\n"
            )
        elif total_results < 3:
            retry_message += (
                "\n💡 **Few results found** - Try:\n"
                "  • Broadening the search with `table='auto'`\n"
                "  • Using more general search terms\n"
            )

        # Add brief note about validation scope
        retry_message += (
            "\n\n*This validation checks search relevance scores. "
            "Other validations handle function names and SQL syntax separately.*"
        )

        # Track that we've validated this query to avoid loops
        if ctx:
            if not hasattr(ctx, "_recent_validation_queries"):
                ctx._recent_validation_queries = set()
            query_key = f"{query[:50]}_{table}"
            ctx._recent_validation_queries.add(query_key)

        logger.info(
            f"Low relevance scores detected, suggesting broader search: {poor_relevance_indicators}"
        )
        raise ModelRetry(retry_message)


async def search_pulseq_knowledge(
    ctx: RunContext[PulsePalDependencies],
    query: str,
    table: str = "auto",
    limit: int = 5,
) -> str:
    """
    Search Pulseq knowledge base with intelligent table selection.

    Choose the most appropriate table(s) based on the search intent:

    **TABLE SELECTION GUIDE:**

    1. "pulseq_sequences" - COMPLETE SEQUENCE IMPLEMENTATIONS
       - Use for: "show me examples", "spiral trajectory", "EPI sequence", specific sequence types
       - Contains: Full MRI sequences with classification metadata (family, trajectory, complexity)
       - Returns: Complete code, dependencies, vendor compatibility, educational value
       - Example queries: "spiral trajectory examples", "TSE sequence", "diffusion imaging"

    2. "api_reference" - PULSEQ FUNCTION DOCUMENTATION
       - Use for: Function syntax, parameters, "how to use makeBlockPulse", API details
       - Contains: Function signatures, parameter specs, return values, usage patterns
       - Returns: Complete API documentation with required/optional parameters
       - Example queries: "makeBlockPulse parameters", "calcDuration usage", "mr.addBlock syntax"

    3. "sequence_chunks" - LOGICAL CODE SECTIONS
       - Use for: Specific implementation patterns, timing calculations, sequence assembly
       - Contains: Categorized code chunks (parameter_definition, rf_gradient_creation, timing_calculations)
       - Returns: Code segment with context, chunk type, MRI concepts demonstrated
       - Example queries: "calculate TE", "gradient spoiling", "sequence assembly loops"

    4. "crawled_code" - AUXILIARY CODE FILES
       - Use for: Helper functions, reconstruction code, vendor conversion tools
       - Contains: Supporting functions, utilities, vendor tools (seq2ceq, TOPPE), reconstruction algorithms
       - Returns: Code content, parent sequences, tool metadata, vendor compatibility
       - Example queries: "GRAPPA reconstruction", "vendor conversion", "helper utilities"

    5. "crawled_docs" - DOCUMENTATION & DATA FILES
       - Use for: Tutorials, guides, data files, vendor documentation, troubleshooting
       - Contains: READMEs, tutorials, CSV/MAT data, vendor guides, configuration files
       - Returns: Document content, metadata, related sequences
       - Example queries: "installation guide", "GE scanner setup", "trajectory data", "vendor troubleshooting"

    **SEARCH STRATEGY:**
    - For sequence examples: Search "pulseq_sequences" first
    - For function help: Search "api_reference" first
    - For implementation details: Search "sequence_chunks" or "pulseq_sequences"
    - For unclear queries: Use "auto" to search across all tables

    **RELEVANCE SCORING:**
    - Results include relevance scores (0-1 scale)
    - If all scores < 0.5, consider broader search or different table
    - Top 3 most relevant results are returned by default

    Special patterns:
    - ID:42 - Direct ID lookup in specified table
    - FILE:writeSpiral.m - Filename-based search

    Args:
        query: User's search query
        table: Table name from above list OR "auto" for automatic selection
        limit: Number of results to return (minimum 5, maximum 10, default 5)
               Note: Automatically adjusted to minimum of 5 to ensure quality results

    Returns:
        JSON with results, relevance scores, and search metadata
    """
    import json

    # Input validation
    if not query or not isinstance(query, str):
        return json.dumps(
            {"error": "Invalid query", "message": "Query must be a non-empty string"}
        )

    # Log very long queries but don't reject them
    # Gemini might generate comprehensive queries
    if len(query) > 1000:
        logger.info(f"Long query received: {len(query)} characters")

    # Enforce minimum limit of 5 for better search quality
    # This ensures users get multiple examples/options to choose from
    if limit < 5:
        logger.info(f"Limit {limit} increased to minimum of 5 for better results")
        limit = 5

    # Cap maximum at 10 to avoid overwhelming responses
    if limit > 10:
        logger.info(f"Limit {limit} capped at maximum of 10")
        limit = 10

    # Get or create RAG service
    if not hasattr(ctx.deps, "rag_v2"):
        ctx.deps.rag_v2 = ModernPulseqRAG()

    # Get detected functions from function detector if available
    detected_functions = None
    if hasattr(ctx.deps, "detected_functions") and ctx.deps.detected_functions:
        detected_functions = ctx.deps.detected_functions
        logger.info(
            f"Using detected functions for enhanced search: {[f['name'] for f in detected_functions]}"
        )

    # Check for special patterns in query
    id_match = re.match(r"^ID:(.+)$", query.strip())
    file_match = re.match(r"^FILE:(.+)$", query.strip())

    # Route to appropriate search method
    if table == "auto":
        # Let the existing source-aware search handle it
        results = await ctx.deps.rag_v2.search_with_source_awareness(
            query=query,
            sources=None,  # Will use all sources
            forced=False,
            source_hints=None,
            detected_functions=detected_functions,
        )
    elif id_match:
        # Handle ID-based lookup
        ids = [id.strip() for id in id_match.group(1).split(",")]
        results = await ctx.deps.rag_v2.search_by_ids(table=table, ids=ids)
    elif file_match:
        # Handle filename-based lookup
        filename = file_match.group(1)
        results = await ctx.deps.rag_v2.search_by_filename(
            table=table, filename=filename
        )
    else:
        # Table-specific search
        results = await ctx.deps.rag_v2.search_table(
            table=table,
            query=query,
            limit=limit,
            detected_functions=detected_functions,
        )

    # VALIDATION: Check relevance scores and prompt for broader search if needed
    _validate_search_relevance(results, table, query, ctx)

    # Log search event if we have context and conversation_context
    if (
        ctx.deps
        and hasattr(ctx.deps, "conversation_context")
        and ctx.deps.conversation_context
    ):
        # Log for table-specific searches
        if "source_table" in results:
            metadata = {
                "table": results.get("source_table", "unknown"),
                "total_results": results.get("total_results", 0),
                "search_type": results.get("search_metadata", {}).get(
                    "search_type", "unknown"
                ),
            }
            if detected_functions:
                metadata["detected_functions"] = [f["name"] for f in detected_functions]

            conversation_logger.log_search_event(
                ctx.deps.conversation_context.session_id,
                "table_specific_rag",
                query,
                results.get("total_results", 0),
                metadata,
            )
        # Log for source-aware searches
        elif "results_by_source" in results:
            metadata = {
                "sources_searched": results.get("search_metadata", {}).get(
                    "sources_searched", []
                ),
                "total_results": results.get("search_metadata", {}).get(
                    "total_results", 0
                ),
            }
            if detected_functions:
                metadata["detected_functions"] = [f["name"] for f in detected_functions]

            conversation_logger.log_search_event(
                ctx.deps.conversation_context.session_id,
                "source_aware_rag",
                query,
                results.get("search_metadata", {}).get("total_results", 0),
                metadata,
            )

    # Return raw JSON results for Gemini to process
    # This follows the specification in docs/RAG-search-plan.md
    return json.dumps(results, indent=2)


def format_sequence_result(result: Dict) -> str:
    """Format pulseq_sequences result for display."""
    parts = []
    parts.append(f"**File:** {result.get('file_name', 'Unknown')}")
    parts.append(f"**Repository:** {result.get('repository', 'Unknown')}")

    if result.get("sequence_family"):
        parts.append(f"**Family:** {result['sequence_family']}")

    if result.get("dependencies"):
        deps = result["dependencies"]
        if deps.get("local"):
            parts.append(f"**Local dependencies:** {', '.join(deps['local'])}")
        if deps.get("pulseq"):
            parts.append(f"**Pulseq functions:** {', '.join(deps['pulseq'])}")

    if result.get("full_code"):
        code = result["full_code"]
        if len(code) > 500:
            code = code[:500] + "\n... (truncated)"
        parts.append(f"```matlab\n{code}\n```")

    return "\n".join(parts) + "\n"


def format_chunk_result(result: Dict) -> str:
    """Format sequence_chunks result for display."""
    parts = []
    parts.append(f"**Chunk Type:** {result.get('chunk_type', 'Unknown')}")
    parts.append(f"**Parent Sequence ID:** {result.get('sequence_id', 'Unknown')}")

    if result.get("mri_concept"):
        parts.append(f"**MRI Concept:** {result['mri_concept']}")

    if result.get("pulseq_functions"):
        parts.append(f"**Functions used:** {', '.join(result['pulseq_functions'])}")

    if result.get("code_content"):
        code = result["code_content"]
        if len(code) > 400:
            code = code[:400] + "\n... (truncated)"
        parts.append(f"```matlab\n{code}\n```")

    return "\n".join(parts) + "\n"


def format_code_result(result: Dict) -> str:
    """Format crawled_code result for display."""
    parts = []
    parts.append(f"**File:** {result.get('file_name', 'Unknown')}")

    if result.get("parent_sequences"):
        parts.append(f"**Used by sequences:** {result['parent_sequences']}")

    if result.get("content"):
        code = result["content"]
        if len(code) > 500:
            code = code[:500] + "\n... (truncated)"
        parts.append(f"```matlab\n{code}\n```")

    return "\n".join(parts) + "\n"


def format_doc_result(result: Dict) -> str:
    """Format crawled_docs result for display."""
    parts = []
    parts.append(f"**Document:** {result.get('resource_uri', 'Unknown')}")

    if result.get("doc_type"):
        parts.append(f"**Type:** {result['doc_type']}")

    if result.get("content"):
        content = result["content"]
        if len(content) > 600:
            content = content[:600] + "\n... (truncated)"
        parts.append(f"Content:\n{content}")

    return "\n".join(parts) + "\n"


def format_api_result(result: Dict) -> str:
    """Format API documentation result for display."""
    parts = []

    # Handle both formats (source-aware and table-specific)
    if "function" in result:
        # Source-aware format
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
    else:
        # Table-specific format
        parts.append(f"**{result.get('function_name', 'Unknown Function')}**")

        if result.get("signature"):
            parts.append(f"Signature: `{result['signature']}`")

        if result.get("description"):
            parts.append(f"Description: {result['description']}")

        if result.get("parameters"):
            params = result["parameters"]
            if isinstance(params, dict):
                if params.get("required"):
                    parts.append("Required parameters:")
                    for p in params["required"]:
                        if isinstance(p, dict):
                            parts.append(
                                f"  - {p.get('name', '')}: {p.get('description', '')}"
                            )
                        else:
                            parts.append(f"  - {p}")
                if params.get("optional"):
                    parts.append("Optional parameters:")
                    for p in params["optional"]:
                        if isinstance(p, dict):
                            parts.append(
                                f"  - {p.get('name', '')}: {p.get('description', '')}"
                            )
                        else:
                            parts.append(f"  - {p}")
            else:
                parts.append(f"Parameters: {params}")

        if result.get("returns"):
            parts.append(f"Returns: {result['returns']}")

    return "\n".join(parts) + "\n"


def format_direct_lookup_result(result: Dict) -> str:
    """Format direct function lookup result for comprehensive display."""
    parts = []

    # Handle both direct lookup format and standard format
    if "function" in result:
        # Standard format from rag_formatters
        func_info = result.get("function", {})
        parts.append(f"**{func_info.get('name', 'Unknown Function')}**")

        if func_info.get("signature"):
            parts.append(f"Signature: `{func_info['signature']}`")

        if func_info.get("description"):
            parts.append(f"Description: {func_info['description']}")

        if func_info.get("calling_pattern"):
            parts.append(f"Usage: `{func_info['calling_pattern']}`")

        # Parameters section
        if result.get("parameters"):
            parts.append(f"\nParameters:\n{result['parameters']}")

        # Returns section
        if result.get("returns"):
            parts.append(f"\nReturns:\n{result['returns']}")

        # Usage examples
        if result.get("usage_examples"):
            parts.append("\n**Usage Examples:**")
            for example in result["usage_examples"]:
                if isinstance(example, str):
                    parts.append(example)
                elif isinstance(example, dict):
                    if example.get("description"):
                        parts.append(f"- {example['description']}")
                    if example.get("code"):
                        parts.append(
                            f"```{example.get('language', 'matlab')}\n{example['code']}\n```"
                        )
    else:
        # Direct database format
        name = result.get("name", result.get("function_name", "Unknown Function"))
        parts.append(f"**{name}**")

        if result.get("signature"):
            parts.append(f"Signature: `{result['signature']}`")

        if result.get("description"):
            parts.append(f"Description: {result['description']}")

        if result.get("calling_pattern"):
            parts.append(f"Usage: `{result['calling_pattern']}`")

        # Handle parameters - could be JSON string or dict
        params = result.get("parameters", {})
        if params:
            if isinstance(params, str):
                try:
                    import json

                    params = json.loads(params)
                except (json.JSONDecodeError, ValueError):
                    pass

            if isinstance(params, dict):
                parts.append("\n**Parameters:**")
                # Check for required/optional structure
                if "required" in params or "optional" in params:
                    if params.get("required"):
                        parts.append("Required:")
                        for p in params["required"]:
                            if isinstance(p, dict):
                                parts.append(
                                    f"• {p.get('name', 'unknown')} ({p.get('type', '')}) - {p.get('description', '')}"
                                )
                            else:
                                parts.append(f"• {p}")
                    if params.get("optional"):
                        parts.append("Optional:")
                        for p in params["optional"]:
                            if isinstance(p, dict):
                                parts.append(
                                    f"• {p.get('name', 'unknown')} ({p.get('type', '')}) - {p.get('description', '')}"
                                )
                            else:
                                parts.append(f"• {p}")
                else:
                    # Direct parameter listing
                    for param_name, param_info in params.items():
                        if isinstance(param_info, dict):
                            param_str = f"• {param_name}"
                            if param_info.get("type"):
                                param_str += f" ({param_info['type']})"
                            if param_info.get("description"):
                                param_str += f" - {param_info['description']}"
                            parts.append(param_str)
                        else:
                            parts.append(f"• {param_name}: {param_info}")
            elif isinstance(params, str):
                parts.append(f"\nParameters:\n{params}")

        # Handle returns
        returns = result.get("returns", {})
        if returns:
            if isinstance(returns, str):
                try:
                    import json

                    returns = json.loads(returns)
                except (json.JSONDecodeError, ValueError):
                    pass

            parts.append("\n**Returns:**")
            if isinstance(returns, dict):
                if returns.get("type"):
                    parts.append(f"Type: {returns['type']}")
                if returns.get("description"):
                    parts.append(f"Description: {returns['description']}")
                if returns.get("fields"):
                    parts.append("Fields:")
                    for field in returns["fields"]:
                        if isinstance(field, dict):
                            parts.append(
                                f"• {field.get('name', '')} - {field.get('description', '')}"
                            )
                        else:
                            parts.append(f"• {field}")
            else:
                parts.append(str(returns))

        # Handle examples
        examples = result.get("usage_examples", [])
        if examples:
            if isinstance(examples, str):
                try:
                    import json

                    examples = json.loads(examples)
                except (json.JSONDecodeError, ValueError):
                    examples = [examples]

            if examples and isinstance(examples, list):
                parts.append("\n**Usage Examples:**")
                for i, example in enumerate(examples, 1):
                    if isinstance(example, dict):
                        if example.get("description"):
                            parts.append(f"Example {i}: {example['description']}")
                        if example.get("code"):
                            parts.append(
                                f"```{example.get('language', 'matlab')}\n{example['code']}\n```"
                            )
                    else:
                        parts.append(f"```matlab\n{example}\n```")

    # Add metadata if available
    metadata = result.get("metadata", {})
    if metadata:
        related = metadata.get("related_functions", [])
        if related:
            parts.append(f"\n**Related Functions:** {', '.join(related)}")

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
                f"Content length: {len(content)} characters (truncated to 10000)",
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


async def find_relevant_tables(
    ctx: RunContext[PulsePalDependencies],
    query: str,
) -> str:
    """
    Find database tables relevant to a natural language query using semantic search.

    This is the first step in database exploration - identifies which tables
    might contain the information needed to answer the query using embedding similarity.

    Args:
        query: Natural language query describing what information is needed

    Returns:
        JSON list of relevant tables with descriptions and similarity scores

    Examples:
        - "list all trajectory types" → ['pulseq_sequences']
        - "function documentation" → ['api_reference']
        - "helper functions" → ['crawled_code']
    """
    import numpy as np
    from pulsepal.embeddings import get_embedding_service

    if not query or not isinstance(query, str):
        return json.dumps(
            {"error": "Invalid query", "message": "Query must be a non-empty string"}
        )

    try:
        # Get RAG service with proper error handling
        rag_service = await get_rag_service(ctx)
        supabase = rag_service.supabase_client.client

        # Get embedding service for query
        embedding_service = get_embedding_service()

        # Create embedding for the query
        logger.debug(f"Creating embedding for query: {query}")
        query_embedding = embedding_service.create_embedding(query)

        # Fetch all table_metadata with embeddings
        try:
            # Test connection first
            supabase.table("table_metadata").select("table_name").limit(1).execute()

            # Get all table metadata with embeddings
            result = (
                supabase.table("table_metadata")
                .select("table_name, description, embedding")
                .execute()
            )

            if not result.data:
                return json.dumps(
                    {
                        "error": "No table metadata found",
                        "message": "Table metadata not initialized",
                    }
                )

        except Exception as conn_error:
            if (
                "connection" in str(conn_error).lower()
                or "timeout" in str(conn_error).lower()
            ):
                return json.dumps(
                    {
                        "error": "Database connection failed",
                        "message": "Unable to connect to knowledge base. Please try again later.",
                        "retry_after": 5,
                    }
                )
            raise

        # Calculate cosine similarity for each table
        scored_tables = []
        query_embedding_np = np.array(query_embedding)

        for table_info in result.data:
            if not table_info.get("embedding"):
                logger.warning(f"No embedding for table {table_info['table_name']}")
                continue

            # Calculate cosine similarity
            # Parse embedding if it's a string (Supabase returns vector as string)
            embedding_data = table_info["embedding"]
            if isinstance(embedding_data, str):
                # Parse the string representation of the vector
                import ast

                embedding_data = ast.literal_eval(embedding_data)
            table_embedding_np = np.array(embedding_data, dtype=np.float32)

            # Cosine similarity = dot product / (norm1 * norm2)
            dot_product = np.dot(query_embedding_np, table_embedding_np)
            norm_query = np.linalg.norm(query_embedding_np)
            norm_table = np.linalg.norm(table_embedding_np)

            if norm_query > 0 and norm_table > 0:
                similarity = dot_product / (norm_query * norm_table)
            else:
                similarity = 0.0

            # Only include tables with meaningful similarity (threshold: 0.3)
            if similarity > 0.3:
                scored_tables.append(
                    {
                        "table_name": table_info["table_name"],
                        "description": table_info["description"],
                        "similarity_score": float(
                            similarity
                        ),  # Convert numpy float to Python float
                    }
                )

        # Sort by similarity score
        scored_tables.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Return top 2 relevant tables with simplified format
        top_tables = scored_tables[:2]  # Only top 2

        # Format response with just table name and description
        results = []
        for table in top_tables:
            results.append(
                {"table": table["table_name"], "description": table["description"]}
            )

        return json.dumps(results, indent=2)

    except Exception as e:
        logger.error(f"Error finding relevant tables: {e}")
        return json.dumps({"error": "Database error", "message": str(e)})


async def get_table_schemas(
    ctx: RunContext[PulsePalDependencies],
    tables: List[str],
) -> str:
    """
    Get detailed schema information for specific tables.

    This is the second step - after identifying relevant tables, get their
    complete schema including columns, data types, and constraints.

    Args:
        tables: List of table names to get schemas for

    Returns:
        JSON with detailed schema information for each table

    Examples:
        - ['api_reference'] → columns, types, constraints for api_reference
        - ['pulseq_sequences', 'sequence_chunks'] → schemas for both tables
    """
    import json

    if not tables or not isinstance(tables, list):
        return json.dumps(
            {"error": "Invalid input", "message": "Tables must be a non-empty list"}
        )

    # Validate table names to prevent SQL injection
    valid_tables = [
        "api_reference",
        "pulseq_sequences",
        "sequence_chunks",
        "crawled_code",
        "crawled_docs",
        "sources",
        "crawled_pages",
    ]

    invalid_tables = [t for t in tables if t not in valid_tables]
    if invalid_tables:
        return json.dumps(
            {
                "error": "Invalid table names",
                "message": f"Unknown tables: {invalid_tables}",
                "valid_tables": valid_tables,
            }
        )

    try:
        schemas = {}

        # Get RAG service with proper error handling
        rag_service = await get_rag_service(ctx)

        try:
            supabase = rag_service.supabase_client.client

            # Test connection first
            supabase.table("table_metadata").select("table_name").limit(1).execute()
        except Exception as conn_error:
            if (
                "connection" in str(conn_error).lower()
                or "timeout" in str(conn_error).lower()
            ):
                return json.dumps(
                    {
                        "error": "Database connection failed",
                        "message": "Unable to connect to knowledge base. Please try again later.",
                        "retry_after": 5,
                    }
                )
            raise

        for table_name in tables:
            # Get metadata description
            metadata_result = (
                supabase.table("table_metadata")
                .select("column_summary, description")
                .eq("table_name", table_name)
                .execute()
            )

            table_description = ""
            column_summary = ""
            if metadata_result.data and len(metadata_result.data) > 0:
                table_description = metadata_result.data[0].get("description", "")
                column_summary = metadata_result.data[0].get("column_summary", "")

            # Parse column summary into structured format
            columns = []
            if column_summary:
                for col_info in column_summary.split(", "):
                    if col_info:
                        columns.append(
                            {
                                "column_name": col_info.strip(),
                                "description": f"Column: {col_info.strip()}",
                            }
                        )

            schemas[table_name] = {
                "table_name": table_name,
                "description": table_description,
                "columns": columns,
                "column_count": len(columns),
            }

        return json.dumps({"tables_requested": tables, "schemas": schemas}, indent=2)

    except Exception as e:
        logger.error(f"Error getting table schemas: {e}")
        return json.dumps({"error": "Database error", "message": str(e)})


async def execute_supabase_query(
    ctx: RunContext[PulsePalDependencies],
    query: Dict[str, Any],
) -> str:
    """
    Execute a flexible Supabase query with validation and error guidance.

    This tool allows constructing any Supabase client query pattern with intelligent
    error handling and correction guidance.

    **Query Structure:**
    ```python
    {
        "table": "api_reference",  # Required: table name
        "select": "*",             # Optional: columns to select (default: "*")
        "filters": [               # Optional: filter conditions
            {"column": "name", "operator": "eq", "value": "makeAdc"},
            {"column": "id", "operator": "gt", "value": 100}
        ],
        "order": {"column": "id", "ascending": true},  # Optional: ordering
        "limit": 10,               # Optional: limit results
        "offset": 0,               # Optional: pagination offset
        "count": false,            # Optional: return count instead of data
        "single": false            # Optional: expect single result
    }
    ```

    **Supported Operators:**
    - `eq`: equals
    - `neq`: not equals
    - `gt`: greater than
    - `gte`: greater than or equal
    - `lt`: less than
    - `lte`: less than or equal
    - `like`: pattern matching (use % for wildcards)
    - `ilike`: case-insensitive pattern matching
    - `in`: value in list
    - `is`: for NULL checks (value should be "null" or "not null")
    - `contains`: for JSONB/array contains
    - `contained_by`: for JSONB/array contained by

    **Examples:**
    ```python
    # Get all functions starting with 'make'
    {
        "table": "api_reference",
        "select": "name, description",
        "filters": [{"column": "name", "operator": "like", "value": "make%"}],
        "limit": 5
    }

    # Get distinct trajectory types
    {
        "table": "pulseq_sequences",
        "select": "trajectory_type",
        "filters": [],
        "order": {"column": "trajectory_type", "ascending": true}
    }
    # Note: For truly distinct values, you may need to process results

    # Count sequences by type
    {
        "table": "pulseq_sequences",
        "select": "sequence_type",
        "count": true
    }
    ```

    **Error Handling:**
    The tool will catch common errors and provide guidance:
    - Table not found → suggests valid tables
    - Column not found → suggests valid columns
    - Type mismatches → explains correct types
    - NULL comparisons → shows correct syntax

    Args:
        query: Supabase query configuration as a dictionary

    Returns:
        JSON string with query results or error with guidance
    """
    import json
    from pydantic_ai import ModelRetry
    from .sql_validator import (
        SupabaseQueryValidator,
        get_table_columns,
        find_similar_columns,
        VALID_TABLES,
    )

    # Validate query structure
    if not isinstance(query, dict):
        raise ModelRetry(
            "Query must be a dictionary. Example:\n"
            '{"table": "api_reference", "select": "*", "limit": 10}'
        )

    if "table" not in query:
        raise ModelRetry(
            "Query must include 'table' field.\n"
            f"Valid tables: {', '.join(VALID_TABLES)}\n"
            "Use find_relevant_tables() to discover appropriate tables."
        )

    table = query["table"]

    # Validate table name
    if table not in VALID_TABLES:
        similar_tables = [t for t in VALID_TABLES if table.lower() in t.lower()]
        raise ModelRetry(
            f"Table '{table}' not found.\n"
            f"Valid tables: {', '.join(VALID_TABLES)}\n"
            + (f"Did you mean: {', '.join(similar_tables)}?" if similar_tables else "")
            + "\nUse find_relevant_tables() to discover appropriate tables."
        )

    try:
        # Get RAG service for Supabase client
        rag_service = await get_rag_service(ctx)
        supabase = rag_service.supabase_client.client

        # Start building query
        query_builder = supabase.table(table)

        # Add select clause with validation
        select_clause = query.get("select", "*")

        # Validate select clause to prevent injection
        if select_clause != "*":
            # Parse comma-separated columns
            columns = [col.strip() for col in select_clause.split(",")]
            for col in columns:
                if not validate_database_identifier(col, "column"):
                    raise ModelRetry(
                        f"Invalid column name in select: '{col}'\n"
                        "Column names must contain only alphanumeric characters and underscores."
                    )

        query_builder = query_builder.select(select_clause)

        # Add filters
        filters = query.get("filters", [])
        for filter_config in filters:
            if not isinstance(filter_config, dict):
                raise ModelRetry(
                    "Each filter must be a dictionary with 'column', 'operator', and 'value'.\n"
                    'Example: {"column": "name", "operator": "eq", "value": "makeAdc"}'
                )

            column = filter_config.get("column")
            operator = filter_config.get("operator", "eq")
            value = filter_config.get("value")

            if not column:
                raise ModelRetry("Filter missing 'column' field")

            # Validate column name to prevent injection
            if not validate_database_identifier(column, "column"):
                raise ModelRetry(
                    f"Invalid column name in filter: '{column}'\n"
                    "Column names must contain only alphanumeric characters and underscores."
                )

            # Apply filter based on operator
            if operator == "eq":
                query_builder = query_builder.eq(column, value)
            elif operator == "neq":
                query_builder = query_builder.neq(column, value)
            elif operator == "gt":
                query_builder = query_builder.gt(column, value)
            elif operator == "gte":
                query_builder = query_builder.gte(column, value)
            elif operator == "lt":
                query_builder = query_builder.lt(column, value)
            elif operator == "lte":
                query_builder = query_builder.lte(column, value)
            elif operator == "like":
                query_builder = query_builder.like(column, value)
            elif operator == "ilike":
                query_builder = query_builder.ilike(column, value)
            elif operator == "in":
                query_builder = query_builder.in_(column, value)
            elif operator == "is":
                # Handle NULL checks
                if value == "null":
                    query_builder = query_builder.is_(column, "null")
                elif value == "not null":
                    query_builder = query_builder.is_(column, "not null")
                else:
                    raise ModelRetry(
                        f"For 'is' operator, value must be 'null' or 'not null', got '{value}'"
                    )
            elif operator == "contains":
                query_builder = query_builder.contains(column, value)
            elif operator == "contained_by":
                query_builder = query_builder.contained_by(column, value)
            else:
                raise ModelRetry(
                    f"Unknown operator '{operator}'.\n"
                    "Valid operators: eq, neq, gt, gte, lt, lte, like, ilike, in, is, contains, contained_by"
                )

        # Add ordering with validation
        if "order" in query:
            order_config = query["order"]
            order_column = order_config.get("column")
            ascending = order_config.get("ascending", True)
            if order_column:
                # Validate order column to prevent injection
                if not validate_database_identifier(order_column, "column"):
                    raise ModelRetry(
                        f"Invalid column name in order: '{order_column}'\n"
                        "Column names must contain only alphanumeric characters and underscores."
                    )
                query_builder = query_builder.order(order_column, desc=not ascending)

        # Add limit
        if "limit" in query:
            limit = query["limit"]
            if not isinstance(limit, int) or limit < 1:
                raise ModelRetry("Limit must be a positive integer")
            query_builder = query_builder.limit(limit)

        # Add offset
        if "offset" in query:
            offset = query["offset"]
            if not isinstance(offset, int) or offset < 0:
                raise ModelRetry("Offset must be a non-negative integer")
            query_builder = query_builder.offset(offset)

        # Handle count queries
        if query.get("count", False):
            query_builder = query_builder.count()

        # Handle single result expectation
        if query.get("single", False):
            query_builder = query_builder.single()

        # Execute query
        result = query_builder.execute()

        # Format successful response
        if result.data is not None:
            response = {
                "success": True,
                "data": result.data,
                "count": result.count if hasattr(result, "count") else None,
                "query_executed": {
                    "table": table,
                    "select": select_clause,
                    "filters": filters,
                    "limit": query.get("limit"),
                    "offset": query.get("offset"),
                },
            }

            # Add helpful metadata
            if isinstance(result.data, list):
                response["row_count"] = len(result.data)

                # If this looks like a distinct values query, extract unique values
                if (
                    select_clause != "*"
                    and "," not in select_clause
                    and len(filters) == 0
                ):
                    # Single column selected with no filters - might be for distinct values
                    # Handle both simple values and arrays
                    values = []
                    for row in result.data:
                        value = row.get(select_clause)
                        if value is not None:
                            # If it's a list/array, add each element
                            if isinstance(value, list):
                                values.extend(value)
                            else:
                                values.append(value)

                    # Try to get unique values (skip if unhashable types)
                    try:
                        unique_values = list(set(values))
                        if len(unique_values) < len(result.data):
                            response["unique_values"] = sorted(unique_values)
                            response["unique_count"] = len(unique_values)
                    except TypeError:
                        # Can't create set from unhashable types, just count raw values
                        response["total_values"] = len(values)
                        response["note"] = "Values contain complex types"

            return json.dumps(response, indent=2, default=str)
        else:
            return json.dumps(
                {
                    "success": True,
                    "data": [],
                    "row_count": 0,
                    "message": "Query executed successfully but returned no results",
                }
            )

    except Exception as e:
        # Safely extract error message
        error_msg = str(e) if e else "Unknown error occurred"

        # Log the actual error for debugging
        logger.error(f"Supabase query error: {error_msg}")

        # Parse the error
        validator = SupabaseQueryValidator()
        parsed_error = validator.parse_postgresql_error(error_msg)

        # Add context for better guidance
        context = {"table": table}

        # If column error, try to get valid columns
        if parsed_error["error_type"] == "column_not_found":
            try:
                valid_columns = await get_table_columns(supabase, table)
                if valid_columns:
                    context["valid_columns"] = valid_columns

                    # Find similar column names
                    missing_col = parsed_error["details"].get("missing_column")
                    if missing_col:
                        similar = find_similar_columns(missing_col, valid_columns)
                        if similar:
                            parsed_error["details"]["similar_columns"] = similar
            except Exception:
                pass  # Continue without column hints

        # Generate guidance
        guidance = validator.generate_error_guidance(parsed_error, context)

        # Log the retry
        logger.warning("🔄 QUERY FAILED - Sending retry guidance to Gemini")
        logger.info(f"  ❌ Error type: {parsed_error.get('error_type', 'unknown')}")
        logger.info(f"  📊 Table: {table}")
        logger.info(f"  ⚠️ Issue: {parsed_error.get('details', {})}")
        logger.debug(f"  Full guidance:\n{guidance}")

        # Raise ModelRetry with guidance
        raise ModelRetry(guidance)


# DEPRECATED: Use execute_supabase_query instead for more flexibility
# Example to get distinct values with the new tool:
# {
#     "table": "pulseq_sequences",
#     "select": "trajectory_type",
#     "order": {"column": "trajectory_type", "ascending": true}
# }
# Then extract unique values from the results
async def _deprecated_get_distinct_values(
    ctx: RunContext[PulsePalDependencies],
    table: str,
    column: str,
    limit: int = 100,
) -> str:
    """
    DEPRECATED: Use execute_supabase_query instead.

    This tool is kept for backward compatibility but should not be used.
    Use execute_supabase_query for more flexible database queries.
    """
    import json

    # Input validation
    if not table or not isinstance(table, str):
        return json.dumps(
            {"error": "Invalid table", "message": "Table must be a non-empty string"}
        )

    if not column or not isinstance(column, str):
        return json.dumps(
            {"error": "Invalid column", "message": "Column must be a non-empty string"}
        )

    # Validate table name to prevent SQL injection
    valid_tables = [
        "api_reference",
        "pulseq_sequences",
        "sequence_chunks",
        "crawled_code",
        "crawled_docs",
        "sources",
        "crawled_pages",
    ]

    if table not in valid_tables:
        return json.dumps(
            {
                "error": "Invalid table name",
                "message": f"Table '{table}' is not valid",
                "valid_tables": valid_tables,
            }
        )

    # Validate column name using our secure validator
    if not validate_database_identifier(column, "column"):
        return json.dumps(
            {
                "error": "Invalid column name",
                "message": "Column name contains invalid characters or is a reserved keyword",
            }
        )

    # Validate limit
    if not isinstance(limit, int) or limit < 1:
        limit = 100
    elif limit > 1000:
        limit = 1000  # Cap at reasonable maximum

    try:
        # Get RAG service with proper error handling
        rag_service = await get_rag_service(ctx)

        try:
            supabase = rag_service.supabase_client.client

            # Test connection first
            supabase.table("table_metadata").select("table_name").limit(1).execute()
        except Exception as conn_error:
            if (
                "connection" in str(conn_error).lower()
                or "timeout" in str(conn_error).lower()
            ):
                return json.dumps(
                    {
                        "error": "Database connection failed",
                        "message": "Unable to connect to knowledge base. Please try again later.",
                        "retry_after": 5,
                    }
                )
            raise

        # Use the new server-side RPC function for efficient DISTINCT query
        result = supabase.rpc(
            "get_distinct_column_values",
            {"p_table_name": table, "p_column_name": column, "p_limit": limit},
        ).execute()

        if not result.data:
            return json.dumps(
                {
                    "table": table,
                    "column": column,
                    "distinct_values": [],
                    "count": 0,
                    "message": "No data found",
                }
            )

        # Parse the RPC function result
        rpc_result = result.data

        # Check if there was an error in the RPC function
        if isinstance(rpc_result, dict) and rpc_result.get("error"):
            return json.dumps(
                {
                    "error": "Database error",
                    "message": rpc_result.get("message", "Unknown error"),
                    "table": table,
                    "column": column,
                }
            )

        # Extract values from the RPC result
        distinct_values = rpc_result.get("values", [])
        total_count = rpc_result.get("count", 0)
        was_limited = rpc_result.get("limited", False)

        return json.dumps(
            {
                "table": table,
                "column": column,
                "distinct_values": distinct_values,
                "count": total_count,
                "limited_to": limit if was_limited else None,
            },
            indent=2,
        )

    except Exception as e:
        logger.error(f"Error getting distinct values from {table}.{column}: {e}")
        return json.dumps(
            {
                "error": "Database error",
                "message": str(e),
                "table": table,
                "column": column,
            }
        )


async def lookup_pulseq_function(
    ctx: RunContext[PulsePalDependencies],
    function_query: str,
) -> str:
    """
    Look up detailed information about Pulseq functions.

    This tool helps verify function names and find the correct function for a task.

    Args:
        function_query: Function name or description of what you want to do
                       Examples: "opts", "rotate", "I need to set system parameters"

    Returns:
        JSON with function details including name, calling pattern, parameters,
        returns, usage examples, and description

    Examples:
        - lookup_pulseq_function("rotate") → finds "rotate3D" function
        - lookup_pulseq_function("makeAdc") → exact match for ADC creation
        - lookup_pulseq_function("gradient limits") → finds "opts" via semantic search

    Security Note:
        Input is validated and sanitized to prevent injection attacks.
    """
    import json
    import re

    # Input validation
    MAX_QUERY_LENGTH = 200
    INVALID_PATTERNS = [
        r"[;<>]",  # SQL/HTML injection characters
        r"--",  # SQL comment
        r"/\*",  # SQL block comment start
        r"\*/",  # SQL block comment end
        r"'",  # Single quote (SQL injection)
        r'"',  # Double quote
        r"`",  # Backtick
        r"\\",  # Backslash
        r"[\x00-\x1f\x7f]",  # Control characters
        r"\b(OR|AND|UNION|SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b",  # SQL keywords
    ]

    if not function_query or not isinstance(function_query, str):
        return json.dumps(
            {"error": "Invalid query", "message": "Query must be a non-empty string"}
        )

    # Length validation
    if len(function_query) > MAX_QUERY_LENGTH:
        return json.dumps(
            {
                "error": "Query too long",
                "message": f"Query must be less than {MAX_QUERY_LENGTH} characters",
            }
        )

    # Sanitize input - strip whitespace
    function_query = function_query.strip()

    # Check for malicious patterns
    for pattern in INVALID_PATTERNS:
        if re.search(pattern, function_query):
            return json.dumps(
                {
                    "error": "Invalid characters",
                    "message": "Query contains invalid characters",
                }
            )

    try:
        # Get RAG service
        rag_service = await get_rag_service(ctx)
        supabase = rag_service.supabase_client.client

        # Phase 1: Try exact match first using Supabase query
        # Check for exact matches in name and calling_pattern columns only
        # Using Supabase's built-in escaping to prevent SQL injection

        # Escape the query for safe use in Supabase filters
        # Supabase client handles parameterization internally

        # Try exact case-sensitive match first
        exact_result = (
            supabase.from_("api_reference")
            .select(
                "name, calling_pattern, description, parameters, returns, usage_examples"
            )
            .or_(f"name.eq.{function_query},calling_pattern.eq.{function_query}")
            .limit(1)
            .execute()
        )

        # If no exact match, try case-insensitive
        if not exact_result.data:
            exact_result = (
                supabase.from_("api_reference")
                .select(
                    "name, calling_pattern, description, parameters, returns, usage_examples"
                )
                .or_(
                    f"name.ilike.{function_query},calling_pattern.ilike.{function_query}"
                )
                .limit(1)
                .execute()
            )

        if exact_result.data and len(exact_result.data) > 0:
            func = exact_result.data[0]
            return json.dumps(
                {
                    "query": function_query,
                    "search_type": "exact_match",
                    "results_found": 1,
                    "functions": [
                        {
                            "function_name": func.get("name"),
                            "calling_pattern": func.get("calling_pattern"),
                            "description": func.get("description"),
                            "parameters": func.get("parameters"),
                            "returns": func.get("returns"),
                            "usage_examples": func.get("usage_examples"),
                        }
                    ],
                    "message": f"Exact match found for function: {func.get('name')}",
                },
                indent=2,
            )

        # Phase 2: Use search_pulseq_knowledge with api_reference table
        # This leverages the full RAG pipeline with BM25 + vector search, RRF fusion, and reranking
        search_result = await search_pulseq_knowledge(
            ctx,
            query=function_query,
            table="api_reference",  # Focus on API reference table only
            limit=2,  # Get top 2 results
        )

        # Parse the search results - search_pulseq_knowledge returns JSON string
        api_functions = []

        try:
            # Try to parse as JSON first
            search_data = json.loads(search_result)

            if "results" in search_data:
                # Extract function information from the results
                for result in search_data["results"][:2]:  # Get top 2
                    # Try to get function name from various fields
                    function_name = (
                        result.get("function_name") or result.get("name") or ""
                    )

                    # If function_name is empty, try to extract from signature
                    if not function_name and result.get("signature"):
                        import re

                        # Extract function name from signature with ReDoS-safe patterns
                        # Limit input length to prevent excessive processing
                        sig = result["signature"][:500] if result["signature"] else ""

                        try:
                            # Use simple, efficient patterns that avoid catastrophic backtracking
                            # Pattern 1: "function name(" - most common case
                            simple_match = re.search(r"function\s+(\w+)\s*\(", sig)
                            if simple_match:
                                function_name = simple_match.group(1)
                            else:
                                # Pattern 2: "function var = name(" or "function [out] = name("
                                # Using [^=]{1,50} to limit backtracking with explicit bounds
                                assign_match = re.search(
                                    r"function\s+[^=]{1,50}=\s*(\w+)\s*\(", sig
                                )
                                if assign_match:
                                    function_name = assign_match.group(1)
                        except Exception as e:
                            # If regex fails, skip extraction
                            logger.warning(
                                f"Failed to extract function name from signature: {e}"
                            )
                            function_name = ""

                    # Build function data
                    func_data = {
                        "function_name": function_name,
                        "calling_pattern": result.get("calling_pattern")
                        or f"mr.{function_name}"
                        if function_name
                        else "",
                        "description": result.get("description") or "",
                        "parameters": result.get("parameters"),
                        "returns": result.get("returns"),
                        "usage_examples": result.get("usage_examples"),
                    }

                    # Only add if we have at least the function name
                    if func_data["function_name"]:
                        api_functions.append(func_data)
        except json.JSONDecodeError:
            # If it's not JSON, check if it contains "No relevant results"
            if "No relevant results found" not in search_result:
                # Try to extract some basic info from the text response
                # This is a fallback in case the format changes
                pass

        # Check if we found any functions from the search
        if api_functions:
            return json.dumps(
                {
                    "query": function_query,
                    "search_type": "hybrid_search",
                    "results_found": len(api_functions),
                    "functions": api_functions,
                    "message": f"Found {len(api_functions)} function(s) matching query: {function_query}",
                },
                indent=2,
            )

        # No results found
        return json.dumps(
            {
                "query": function_query,
                "search_type": "no_results",
                "results_found": 0,
                "functions": [],
                "message": f"No Pulseq functions found matching '{function_query}'",
            },
            indent=2,
        )

    except Exception as e:
        # Log detailed error for debugging (not exposed to user)
        logger.error(f"Error looking up function '{function_query}': {e}")

        # Return sanitized error messages to prevent information disclosure
        # Check if it's a connection error
        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            return json.dumps(
                {
                    "error": "Service unavailable",
                    "message": "Unable to connect to knowledge base. Please try again later.",
                    "query": function_query,
                }
            )

        # Generic error message - don't expose internal details
        return json.dumps(
            {
                "error": "Lookup error",
                "message": "An error occurred while searching for the function. Please try again.",
                "query": function_query,
            }
        )
