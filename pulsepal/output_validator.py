"""
Output validator for PulsePal agent responses.
Detects and validates Pulseq functions in Gemini's output.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic_ai import ModelRetry, RunContext

from .dependencies import PulsePalDependencies
from .function_index import (
    COMMON_HALLUCINATIONS,
    MATLAB_FUNCTIONS,
    get_correct_namespace,
)
from .tag_validator import (
    get_session_whitelist,
    should_skip_function_validation,
)

logger = logging.getLogger(__name__)

# Maximum retries for validation
MAX_VALIDATION_RETRIES = 3
MAX_OUTPUT_LENGTH = 100000  # Maximum output length to prevent ReDoS

# Track retry count per session with thread-safe access
_retry_counts: Dict[str, int] = {}
_retry_timestamps: Dict[str, datetime] = {}
_retry_lock = asyncio.Lock()
RETRY_EXPIRATION_HOURS = 24

# Pre-compiled regex patterns for better performance
PULSEQ_PATTERNS = [
    re.compile(r"\b(mr|seq|tra|eve|opt)\.([A-Za-z]\w*)"),
    re.compile(r"\b(mr)\.aux\.([A-Za-z]\w*)"),
    re.compile(r"\b(mr)\.aux\.quat\.([A-Za-z]\w*)"),
    re.compile(r"\b(mr)\.Siemens\.([A-Za-z]\w*)"),
    re.compile(r"\b(mrMusic)\.([A-Za-z]\w*)"),
]
CODE_BLOCK_PATTERN = re.compile(r"```(?:matlab|python|m)?\n(.*?)```", re.DOTALL)
INLINE_CODE_PATTERN = re.compile(r"`([^`]+)`")


def extract_pulseq_functions(text: str) -> Set[Tuple[str, str, int]]:
    """
    Extract all Pulseq function calls from text with input validation.

    Extracts from:
    1. Code blocks (```matlab or ```python)
    2. Inline code mentions (e.g., `mr.makeAdc()`)
    3. Plain text mentions of functions

    Args:
        text: The complete response text

    Returns:
        Set of tuples: (function_call, context, line_number)
        e.g., {("mr.makeAdc", "in code block", 15), ("seq.write", "inline mention", 42)}
    """
    # Input validation to prevent ReDoS
    if len(text) > MAX_OUTPUT_LENGTH:
        logger.warning(
            f"Output too long ({len(text)} chars), truncating to {MAX_OUTPUT_LENGTH}"
        )
        text = text[:MAX_OUTPUT_LENGTH]

    functions = set()

    try:
        # Extract from code blocks
        code_blocks = CODE_BLOCK_PATTERN.finditer(text)

        for block_match in code_blocks:
            code_content = block_match.group(1)
            start_line = text[: block_match.start()].count("\n") + 1

            # Search for functions in code block
            for line_offset, line in enumerate(code_content.split("\n")):
                # Skip comments
                if line.strip().startswith("%") or line.strip().startswith("#"):
                    continue

                for pattern in PULSEQ_PATTERNS:
                    matches = pattern.finditer(line)
                    for match in matches:
                        full_match = match.group(0)
                        line_num = start_line + line_offset + 1
                        functions.add((full_match, "code_block", line_num))

        # Extract from inline code mentions (e.g., `mr.makeAdc`)
        inline_matches = INLINE_CODE_PATTERN.finditer(text)

        for match in inline_matches:
            inline_content = match.group(1)
            line_num = text[: match.start()].count("\n") + 1

            for pattern in PULSEQ_PATTERNS:
                func_matches = pattern.finditer(inline_content)
                for func_match in func_matches:
                    functions.add((func_match.group(0), "inline_code", line_num))

        # Extract from plain text (outside code blocks and inline code)
        # Remove code blocks and inline code first
        text_without_code = CODE_BLOCK_PATTERN.sub("", text)
        text_without_code = INLINE_CODE_PATTERN.sub("", text_without_code)

        for line_num, line in enumerate(text_without_code.split("\n"), 1):
            for pattern in PULSEQ_PATTERNS:
                matches = pattern.finditer(line)
                for match in matches:
                    functions.add((match.group(0), "plain_text", line_num))

    except Exception as e:
        logger.error(f"Error extracting Pulseq functions: {e}", exc_info=True)
        # Return empty set on error to avoid breaking validation
        return set()

    return functions


def validate_pulseq_function(
    function_call: str, session_whitelist: Set[str] = None
) -> Dict[str, Any]:
    """
    Validate a single Pulseq function call.

    Args:
        function_call: Function call like "mr.makeAdc" or "seq.write"
        session_whitelist: Set of custom functions to allow for this session

    Returns:
        Dictionary with validation results:
        - is_valid: bool
        - error_type: "not_found" | "wrong_namespace" | "wrong_capitalization" | None
        - correct_form: str or None
        - explanation: str
    """
    # Check if function is in session whitelist
    if session_whitelist and function_call in session_whitelist:
        return {
            "is_valid": True,
            "error_type": None,
            "correct_form": function_call,
            "explanation": "Function is whitelisted from retrieved examples.",
        }
    # Parse the function call
    parts = function_call.split(".")

    if len(parts) == 1:
        # Just a function name without namespace
        func_name = parts[0]
        namespace = None
    else:
        # Has namespace
        func_name = parts[-1]
        namespace = ".".join(parts[:-1])

    # Build set of all valid functions
    all_functions = set()
    all_functions.update(MATLAB_FUNCTIONS.get("direct_calls", set()))
    for methods in MATLAB_FUNCTIONS.get("class_methods", {}).values():
        all_functions.update(methods)
    all_functions.update(MATLAB_FUNCTIONS.get("mr_aux_functions", set()))
    all_functions.update(MATLAB_FUNCTIONS.get("mr_aux_quat_functions", set()))
    all_functions.update(MATLAB_FUNCTIONS.get("mr_siemens_functions", set()))
    all_functions.update(MATLAB_FUNCTIONS.get("mrMusic_functions", set()))

    # Check if function exists (case-insensitive)
    func_exists = False
    correct_func_name = None
    for valid_func in all_functions:
        if valid_func.lower() == func_name.lower():
            func_exists = True
            correct_func_name = valid_func
            break

    if not func_exists:
        # Check if it's a known hallucination
        if func_name in COMMON_HALLUCINATIONS:
            correct = COMMON_HALLUCINATIONS[func_name]
            if correct:
                correct_namespace = get_correct_namespace(correct)
                correct_form = (
                    f"{correct_namespace}.{correct}" if correct_namespace else correct
                )
                return {
                    "is_valid": False,
                    "error_type": "not_found",
                    "correct_form": correct_form,
                    "explanation": f"'{func_name}' is a common misconception. The correct function is '{correct}'.",
                }
            else:
                return {
                    "is_valid": False,
                    "error_type": "not_found",
                    "correct_form": None,
                    "explanation": f"'{func_name}' does not exist in Pulseq.",
                }
        else:
            return {
                "is_valid": False,
                "error_type": "not_found",
                "correct_form": None,
                "explanation": f"'{func_name}' is not a valid Pulseq function.",
            }

    # Function exists, check capitalization
    if correct_func_name != func_name:
        correct_namespace = get_correct_namespace(correct_func_name)
        correct_form = (
            f"{correct_namespace}.{correct_func_name}"
            if correct_namespace
            else correct_func_name
        )
        return {
            "is_valid": False,
            "error_type": "wrong_capitalization",
            "correct_form": correct_form,
            "explanation": f"Incorrect capitalization: '{func_name}' should be '{correct_func_name}'.",
        }

    # Check namespace
    correct_namespace = get_correct_namespace(func_name)

    if correct_namespace is None:
        # Function not in our namespace map (shouldn't happen if it exists)
        return {
            "is_valid": True,
            "error_type": None,
            "correct_form": function_call,
            "explanation": "Function appears valid.",
        }

    if correct_namespace == "":
        # No namespace needed (constructor or standalone)
        if namespace:
            return {
                "is_valid": False,
                "error_type": "wrong_namespace",
                "correct_form": func_name,
                "explanation": f"'{func_name}' should be called without a namespace.",
            }
    else:
        # Namespace required
        if namespace != correct_namespace:
            correct_form = f"{correct_namespace}.{func_name}"
            return {
                "is_valid": False,
                "error_type": "wrong_namespace",
                "correct_form": correct_form,
                "explanation": f"Wrong namespace: '{function_call}' should be '{correct_form}'.",
            }

    # All checks passed
    return {
        "is_valid": True,
        "error_type": None,
        "correct_form": function_call,
        "explanation": "Function is valid.",
    }


def find_similar_function(func_name: str, max_distance: int = 3) -> Optional[str]:
    """
    Find similar function names using Levenshtein distance.

    Args:
        func_name: The incorrect function name
        max_distance: Maximum edit distance to consider (default: 3)

    Returns:
        The most similar function name if found, None otherwise
    """
    # Build complete list of valid functions
    all_functions = set()
    all_functions.update(MATLAB_FUNCTIONS.get("direct_calls", set()))
    for methods in MATLAB_FUNCTIONS.get("class_methods", {}).values():
        all_functions.update(methods)
    all_functions.update(MATLAB_FUNCTIONS.get("mr_aux_functions", set()))
    all_functions.update(MATLAB_FUNCTIONS.get("mr_aux_quat_functions", set()))
    all_functions.update(MATLAB_FUNCTIONS.get("mr_siemens_functions", set()))
    all_functions.update(MATLAB_FUNCTIONS.get("mrMusic_functions", set()))

    # Simple Levenshtein distance implementation
    def levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # j+1 instead of j since previous_row and current_row are one character longer
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    # Find the closest match
    best_match = None
    best_distance = max_distance + 1

    for valid_func in all_functions:
        distance = levenshtein(func_name.lower(), valid_func.lower())
        if distance < best_distance:
            best_distance = distance
            best_match = valid_func

    return best_match if best_distance <= max_distance else None


def build_validation_error_message(
    invalid_functions: List[Tuple[str, Dict]], session_id: str
) -> str:
    """
    Build a clear error message that informs Gemini what's wrong and provides tools to fix it.
    Follows agentic AI principles - inform and provide tools, don't assume intent.

    Args:
        invalid_functions: List of (function_call, validation_result) tuples
        session_id: Session ID for retry tracking

    Returns:
        Formatted error message for ModelRetry
    """
    # Group errors by type
    not_found = []
    wrong_namespace = []
    wrong_capitalization = []

    for func_call, validation in invalid_functions:
        error_type = validation["error_type"]
        if error_type == "not_found":
            not_found.append((func_call, validation))
        elif error_type == "wrong_namespace":
            wrong_namespace.append((func_call, validation))
        elif error_type == "wrong_capitalization":
            wrong_capitalization.append((func_call, validation))

    # Build error message
    message_parts = ["❌ **Invalid Pulseq Functions Detected**\n"]

    # Report functions that don't exist with fuzzy matching suggestions
    if not_found:
        message_parts.append("**Functions that don't exist:**")
        for func_call, validation in not_found:
            # Extract just the function name (remove namespace)
            func_name = func_call.split(".")[-1]
            message_parts.append(f"• `{func_call}` - {validation['explanation']}")

            # Check for known corrections first
            if validation.get("correct_form"):
                message_parts.append(
                    f"  → Known correction: `{validation['correct_form']}`"
                )
            else:
                # Try fuzzy matching for suggestions
                similar = find_similar_function(func_name)
                if similar:
                    correct_namespace = get_correct_namespace(similar)
                    if correct_namespace:
                        message_parts.append(
                            f"  → Did you mean: `{correct_namespace}.{similar}`?"
                        )
                    else:
                        message_parts.append(f"  → Did you mean: `{similar}`?")
        message_parts.append("")

    # Handle wrong namespace - these have clear corrections
    if wrong_namespace:
        message_parts.append("**Functions with incorrect namespace:**")
        for func_call, validation in wrong_namespace:
            message_parts.append(
                f"• `{func_call}` → Should be `{validation['correct_form']}`"
            )
        message_parts.append("")

    # Handle wrong capitalization - also have clear corrections
    if wrong_capitalization:
        message_parts.append("**Functions with incorrect capitalization:**")
        for func_call, validation in wrong_capitalization:
            message_parts.append(
                f"• `{func_call}` → Should be `{validation['correct_form']}`"
            )
        message_parts.append("")

    # Provide tools and guidance based on retry count
    retry_count = _retry_counts.get(session_id, 0)

    message_parts.append("**💡 Use the function lookup tool to verify:**\n")

    # Show specific examples for the invalid functions
    unique_funcs = set()
    for func_call, _ in invalid_functions:
        func_name = func_call.split(".")[-1]
        if func_name not in unique_funcs:
            unique_funcs.add(func_name)
            message_parts.append(f'`lookup_pulseq_function("{func_name}")`')

    message_parts.append("\nThis tool will:")
    message_parts.append("• Find the correct function name if it exists")
    message_parts.append("• Show complete details: parameters, returns, usage examples")
    message_parts.append("• Suggest similar functions if no exact match\n")

    message_parts.append("**Alternative search methods:**\n")

    message_parts.append("1. **Search by intent:**")
    message_parts.append('   `lookup_pulseq_function("function to rotate gradients")`')
    message_parts.append("   Finds functions matching your description\n")

    message_parts.append("2. **Browse function documentation:**")
    message_parts.append(
        "   `search_pulseq_knowledge(query='[concept]', table='api_reference', limit=5)`"
    )
    message_parts.append("   Returns multiple related functions\n")

    if retry_count == 0:
        message_parts.append("**Tip:** Search broadly to explore what's available:")
        message_parts.append(
            "• Search for common terms: 'make' (finds creation functions)"
        )
        message_parts.append(
            "• Search by concept: 'gradient', 'pulse', 'ADC', 'timing'"
        )
        message_parts.append(
            "• Combine terms: 'trapezoid gradient area' for specific needs"
        )
        message_parts.append("• Increase limit to see more: `limit=20` or `limit=30`\n")
    elif retry_count == 1:
        message_parts.append(
            "**Note:** Some operations require multiple Pulseq functions working together."
        )
        message_parts.append(
            "Consider searching for broader concepts or checking sequence examples.\n"
        )
    else:
        message_parts.append("**Note:** The operation you're looking for might:")
        message_parts.append("• Require a sequence of multiple functions")
        message_parts.append("• Not exist as a single function in Pulseq")
        message_parts.append("• Need to be implemented manually\n")

    message_parts.append(
        f"*Validation retry {retry_count + 1} of {MAX_VALIDATION_RETRIES}*"
    )

    return "\n".join(message_parts)


async def validate_pulseq_output(
    ctx: RunContext[PulsePalDependencies], output: str
) -> str:
    """
    Output validator for PulsePal agent responses with thread-safe retry management.
    Validates Pulseq functions and prompts for correction if needed.

    Args:
        ctx: Run context with dependencies
        output: The agent's response text

    Returns:
        The validated output

    Raises:
        ModelRetry: If invalid functions are found (up to MAX_VALIDATION_RETRIES times)
    """
    # Get session ID for retry tracking
    session_id = (
        ctx.deps.conversation_context.session_id
        if ctx.deps.conversation_context
        else "default"
    )

    # Check if we should skip function validation (all code is retrieved examples)
    if await should_skip_function_validation(session_id):
        logger.info(
            f"Skipping function validation for session {session_id} (retrieved examples only)"
        )
        return output

    # Thread-safe retry count management
    async with _retry_lock:
        # Initialize retry count for this session if needed
        if session_id not in _retry_counts:
            _retry_counts[session_id] = 0
            _retry_timestamps[session_id] = datetime.now()

        # Clean up expired sessions
        current_time = datetime.now()
        expired_sessions = [
            sid
            for sid, timestamp in _retry_timestamps.items()
            if current_time - timestamp > timedelta(hours=RETRY_EXPIRATION_HOURS)
        ]
        for sid in expired_sessions:
            del _retry_counts[sid]
            del _retry_timestamps[sid]
            logger.debug(f"Cleaned up expired retry counter for session {sid}")

        # Check if we've exceeded retry limit
        if _retry_counts[session_id] >= MAX_VALIDATION_RETRIES:
            logger.warning(f"Exceeded validation retry limit for session {session_id}")
            # Reset counter and allow the response through with a warning
            _retry_counts[session_id] = 0
            return (
                output
                + "\n\n*Note: Some functions may not be valid Pulseq functions. Please verify before use.*"
            )

    # Extract all Pulseq functions from the output
    functions = extract_pulseq_functions(output)

    if not functions:
        # No Pulseq functions found, output is valid
        _retry_counts[session_id] = 0  # Reset counter on success
        return output

    # Get session whitelist for custom functions
    session_whitelist = await get_session_whitelist(session_id)

    # Validate each function
    invalid_functions = []
    for func_call, context, line_num in functions:
        validation = validate_pulseq_function(func_call, session_whitelist)
        if not validation["is_valid"]:
            invalid_functions.append((func_call, validation))
            logger.info(
                f"Invalid function found: {func_call} at line {line_num} ({context})"
            )

    if not invalid_functions:
        # All functions are valid - reset counter on success
        async with _retry_lock:
            _retry_counts[session_id] = 0
            _retry_timestamps[session_id] = datetime.now()
        return output

    # Build error message with tools and guidance
    error_message = build_validation_error_message(invalid_functions, session_id)

    # Thread-safe increment of retry counter
    async with _retry_lock:
        _retry_counts[session_id] += 1
        retry_count = _retry_counts[session_id]
        _retry_timestamps[session_id] = datetime.now()  # Update timestamp

    logger.warning(
        f"🔄 VALIDATION FAILED for session {session_id}, retry {retry_count}/{MAX_VALIDATION_RETRIES}"
    )
    logger.info(f"  ❌ Invalid functions detected: {[f[0] for f in invalid_functions]}")
    logger.info(f"  📝 Original message preview: {output[:200]}...")
    logger.info("  🔧 Sending correction guidance to Gemini...")
    logger.debug(f"  Full error message:\n{error_message}")

    raise ModelRetry(error_message)


# Function to reset retry counter (useful for new conversations)
async def reset_validation_retries(session_id: str):
    """Reset the validation retry counter for a session in a thread-safe manner."""
    async with _retry_lock:
        if session_id in _retry_counts:
            del _retry_counts[session_id]
        if session_id in _retry_timestamps:
            del _retry_timestamps[session_id]
        logger.debug(f"Reset validation retry counter for session {session_id}")
