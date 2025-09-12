"""
Combined validator for PulsePal that ensures proper ordering and state management.
Simplified to only perform function validation without tag requirements.
"""

import logging

from pydantic_ai import ModelRetry, RunContext

from .dependencies import PulsePalDependencies
from .validation_state import get_validation_state_manager
from .output_validator import (
    extract_pulseq_functions,
    validate_pulseq_function,
    build_validation_error_message,
    MAX_VALIDATION_RETRIES,
)

logger = logging.getLogger(__name__)


async def combined_output_validator(
    ctx: RunContext[PulsePalDependencies], output: str
) -> str:
    """
    Simplified validator that only performs function validation.
    No longer requires code blocks to be tagged.

    Args:
        ctx: Run context with dependencies
        output: The agent's response

    Returns:
        Validated output

    Raises:
        ModelRetry: If validation fails (up to MAX_VALIDATION_RETRIES times)
    """
    # Get session ID
    session_id = (
        ctx.deps.conversation_context.session_id
        if ctx.deps.conversation_context
        else "default"
    )

    # Get or create validation state for this session
    state_manager = get_validation_state_manager()
    validation_state = await state_manager.get_or_create_state(session_id)

    # Extract all Pulseq functions for validation
    functions = extract_pulseq_functions(output)

    if not functions:
        # No functions to validate - successful completion
        validation_state.reset_retry_counters()
        return output

    # Get whitelist from validation state for custom functions
    whitelist = validation_state.get_whitelist()

    # Validate each function
    invalid_functions = []
    for func_call, context, line_num in functions:
        validation = validate_pulseq_function(func_call, whitelist)
        if not validation["is_valid"]:
            invalid_functions.append((func_call, validation))
            logger.info(f"Invalid function: {func_call} at line {line_num}")

    if invalid_functions:
        # Increment retry counter first
        current_retry = validation_state.increment_func_retry()

        # Check if we've exceeded the maximum retries (after incrementing)
        if current_retry > MAX_VALIDATION_RETRIES:
            logger.warning(
                f"Exceeded function validation retry limit ({MAX_VALIDATION_RETRIES}) for session {session_id}"
            )
            # Reset counters and pass through without modification
            validation_state.reset_retry_counters()
            logger.info("Passing through response despite invalid functions")

            # Return output as-is without adding any notes
            return output

        # Still within retry limit, request correction
        logger.info(
            f"Invalid functions found (retry {current_retry}/{MAX_VALIDATION_RETRIES})"
        )

        # Build error message (this already includes retry count in the message)
        error_message = build_validation_error_message(invalid_functions, session_id)

        raise ModelRetry(error_message)

    # All validation successful - reset retry counters
    validation_state.reset_retry_counters()

    return output
