"""
Simplified tools for Pulsepal agent with modern RAG service v2.

Provides tools that use the simplified retrieval-only RAG service,
leaving all intelligence to the LLM.
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from pydantic_ai import RunContext

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


async def search_pulseq_knowledge(
    ctx: RunContext[PulsePalDependencies],
    query: str,
    table: str = "auto",
    limit: int = 3,
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
        limit: Maximum results (default 3, max 10)

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


async def verify_function_namespace(
    ctx: RunContext[PulsePalDependencies],
    function_call: str,
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
    return f"✅ `{function_call}` uses the correct namespace."


async def validate_pulseq_function(
    ctx: RunContext[PulsePalDependencies],
    function_name: str,
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
    response = [f"❌ {function_name} is not valid"]

    if result.get("correct_form"):
        response.append(f"✅ Use: {result['correct_form']}")

    if result.get("explanation"):
        response.append(f"ℹ️ {result['explanation']}")

    if result.get("suggestions"):
        response.append(f"💡 Suggestions: {', '.join(result['suggestions'])}")

    return "\n".join(response)


async def validate_code_block(
    ctx: RunContext[PulsePalDependencies],
    code: str,
    language: str = "matlab",
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


async def get_distinct_values(
    ctx: RunContext[PulsePalDependencies],
    table: str,
    column: str,
    limit: int = 100,
) -> str:
    """
    Get all distinct values from a specific column in a table.

    This tool enables complete enumeration of categorical data, solving the
    "list all X" problem that text search cannot handle effectively.

    Args:
        table: Table name to query
        column: Column name to get distinct values from
        limit: Maximum number of distinct values to return (default 100)

    Returns:
        JSON with complete list of distinct values and count

    Examples:
        - table='pulseq_sequences', column='trajectory_type' → all trajectory types
        - table='api_reference', column='class_name' → all class names
        - table='pulseq_sequences', column='sequence_family' → all sequence families
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
