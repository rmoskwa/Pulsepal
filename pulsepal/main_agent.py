"""
Main Pulsepal agent with modern RAG service v2.

Simplified version that lets the LLM handle all intelligence,
while RAG just retrieves documents.
"""

import logging
import uuid

from pydantic_ai import Agent

from .dependencies import PulsePalDependencies, get_session_manager
from .gemini_patch import GeminiRecitationError
from .providers import get_llm_model

logger = logging.getLogger(__name__)

# Simplified system prompt - trust Gemini's intelligence more
PULSEPAL_SYSTEM_PROMPT = """You are PulsePal, an expert MRI physics and Pulseq programming assistant.

## Expertise
- Deep MRI physics knowledge (built-in)
- Pulseq framework (MATLAB/Python)
- Debugging sequences and artifacts
- Educational support for researchers

## Knowledge Sources
You have built-in MRI physics knowledge AND access to Pulseq-specific documentation via search_pulseq_knowledge:
- API documentation (authoritative function specs)
- Official sequence examples (validated implementations)
- Community discussions and solutions

Search strategically - not everything needs a lookup. Use your judgment.

## Tools & Validation
- `validate_pulseq_function`: Check function correctness when needed
- Use validation for: function errors, suspicious patterns (seq.calcKspace, mr.write), or explicit requests
- Detected functions are hints, not mandates - you decide what needs validation

## Response Guidelines
- Default to MATLAB unless Python specified
- Synthesize multiple sources when available
- Never fabricate functions when coding - validate if unsure

Trust your intelligence to balance thoroughness with efficiency. Remember: You have both MRI domain expertise and access to Pulseq-specific documentation.
Use both to provide comprehensive, accurate assistance."""

# Create Pulsepal agent
pulsepal_agent = Agent(
    get_llm_model(),
    deps_type=PulsePalDependencies,
    system_prompt=PULSEPAL_SYSTEM_PROMPT,
)


# Import and register tools after agent creation
def _register_tools():
    """Register enhanced tools with the pulsepal agent."""
    from . import tools

    # Register modern tools with validation
    pulsepal_agent.tool(tools.search_pulseq_knowledge)
    pulsepal_agent.tool(tools.verify_function_namespace)
    pulsepal_agent.tool(
        tools.validate_pulseq_function,
    )  # Critical for hallucination prevention
    pulsepal_agent.tool(tools.validate_code_block)  # Validate entire code blocks

    # Set the agent reference in tools module
    tools.pulsepal_agent = pulsepal_agent

    logger.info("Modern tools registered with Pulsepal agent")


# Register tools on module import
_register_tools()

# Cached semantic router instance (singleton pattern)
_semantic_router_instance = None


def get_semantic_router():
    """Get or create a singleton SemanticRouter instance.

    This ensures the 80MB model is only loaded once, not on every request.
    """
    global _semantic_router_instance

    # Skip semantic router in CLI mode or when disabled
    import os

    if os.getenv("DISABLE_SEMANTIC_ROUTER", "false").lower() == "true":
        logger.info("Semantic router disabled via environment variable")
        return None

    if _semantic_router_instance is None:
        try:
            from .semantic_router import SemanticRouter

            logger.info("Initializing semantic router (one-time load)...")
            _semantic_router_instance = SemanticRouter(lazy_load=False)
            logger.info("Semantic router initialized and cached")
        except Exception as e:
            logger.warning(f"Failed to initialize semantic router: {e}")
            # Return None to indicate router is not available
            return None

    return _semantic_router_instance


async def create_pulsepal_session(
    session_id: str = None,
    query: str = None,
) -> tuple[str, PulsePalDependencies]:
    """
    Create a new Pulsepal session with dependencies.

    Args:
        session_id: Optional session ID to reuse
        query: Optional query for semantic routing

    Returns:
        Tuple of (session_id, dependencies)
    """
    # Generate or reuse session ID
    if not session_id:
        session_id = str(uuid.uuid4())

    # Get session manager
    session_manager = get_session_manager()

    # Create or get session (get_session handles creation if needed)
    conversation_context = session_manager.get_session(session_id)

    # Create dependencies
    deps = PulsePalDependencies(
        conversation_context=conversation_context,
        session_manager=session_manager,
    )

    # Initialize RAG services (now simplified)
    await deps.initialize_rag_services()

    # Apply function detection if query provided (no routing restrictions)
    if query:
        try:
            router = get_semantic_router()
            if router is None:
                logger.warning(
                    "Semantic router not available, skipping function detection"
                )
            else:
                routing_decision = router.classify_query(query)

                # Only use detected functions as hints, not for routing decisions
                if routing_decision.detected_functions:
                    deps.detected_functions = routing_decision.detected_functions
                    deps.validation_errors = routing_decision.validation_errors

                    logger.info(
                        f"Function detector found {len(routing_decision.detected_functions)} function(s): "
                        f"{[f['name'] for f in routing_decision.detected_functions]}",
                    )

                    if routing_decision.validation_errors:
                        logger.warning(
                            f"Validation errors detected: {routing_decision.validation_errors}"
                        )

                # Log the detection but don't restrict Gemini's choices
                logger.debug(
                    "Function detection complete. Gemini will decide search strategy."
                )

        except ImportError as e:
            logger.warning(f"Function detector not available: {e}")
            # Continue without function detection
        except Exception as e:
            logger.error(f"Function detection failed: {e}")
            # Continue without function detection - fail gracefully

    logger.info(f"Created Pulsepal session: {session_id}")

    return session_id, deps


async def run_pulsepal_query(
    query: str,
    session_id: str = None,
    temperature: float = 0.1,
) -> tuple[str, str]:
    """
    Run a query through the Pulsepal agent.

    Args:
        query: User query
        session_id: Optional session ID for context
        temperature: LLM temperature setting

    Returns:
        Tuple of (session_id, response)
    """
    try:
        # Create or get session with semantic routing
        session_id, deps = await create_pulsepal_session(session_id, query)

        # Add query to conversation history
        deps.conversation_context.add_conversation("user", query)

        # Run the agent
        result = await pulsepal_agent.run(
            query,
            deps=deps,
            model_settings={"temperature": temperature},
        )

        # Extract response - use result.output for pydantic-ai
        response = result.output if hasattr(result, "output") else str(result)

        # Add response to conversation history
        deps.conversation_context.add_conversation("assistant", response)

        return session_id, response

    except GeminiRecitationError as e:
        logger.warning(f"Recitation error: {e}")
        # Fallback response for recitation errors
        return session_id, (
            "I encountered an issue with content generation. "
            "Please try rephrasing your question or asking for a different example."
        )
    except Exception as e:
        # Log the full error with traceback for debugging
        import traceback

        logger.error(f"Error running Pulsepal query: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")

        # Check for specific error types and provide helpful feedback
        error_msg = str(e).lower()

        if "503" in error_msg or "service unavailable" in error_msg:
            logger.error("Gemini API returned 503 Service Unavailable")
            return session_id, (
                "⚠️ **Gemini API Service Unavailable (503)**\n\n"
                "The AI service is temporarily unavailable. This usually means:\n"
                "- The service is experiencing high load\n"
                "- Temporary maintenance is occurring\n\n"
                "Please try again in a few moments. If the issue persists, "
                "check the Gemini API status page.\n\n"
                f"Error details: {e!s}"
            )
        if "rate limit" in error_msg or "429" in error_msg:
            logger.error("Rate limit exceeded for Gemini API")
            return session_id, (
                "⚠️ **Rate Limit Exceeded**\n\n"
                "The API rate limit has been reached. Please wait a moment before trying again.\n\n"
                f"Error details: {e!s}"
            )
        if "timeout" in error_msg:
            logger.error("Request timeout")
            return session_id, (
                "⚠️ **Request Timeout**\n\n"
                "The request took too long to process. Try:\n"
                "- Breaking your query into smaller parts\n"
                "- Being more specific in your question\n\n"
                f"Error details: {e!s}"
            )
        if "connection" in error_msg or "network" in error_msg:
            logger.error("Network/connection error")
            return session_id, (
                "⚠️ **Connection Error**\n\n"
                "Unable to connect to the AI service. Please check:\n"
                "- Your internet connection\n"
                "- Firewall/proxy settings\n"
                "- API endpoint availability\n\n"
                f"Error details: {e!s}"
            )
        # For unexpected errors, provide the actual error message
        # This ensures we don't hide important debugging information
        logger.error(f"Unexpected error type: {type(e).__name__}")
        return session_id, (
            f"❌ **Error: {type(e).__name__}**\n\n"
            f"An unexpected error occurred:\n{e!s}\n\n"
            "This error has been logged for debugging. "
            "Please try rephrasing your query or contact support if the issue persists."
        )


def apply_semantic_routing(query: str, deps: PulsePalDependencies) -> None:
    """
    Apply function detection to a query and update dependencies.
    This is a standalone function for use by Chainlit.
    No longer restricts search decisions - only provides function hints.

    Args:
        query: The user query to analyze for functions
        deps: Dependencies object to update with detected functions
    """
    try:
        router = get_semantic_router()
        if router is None:
            logger.warning("Semantic router not available, skipping function detection")
            return

        routing_decision = router.classify_query(query)

        # Only use detected functions as hints, not for routing decisions
        if routing_decision.detected_functions:
            deps.detected_functions = routing_decision.detected_functions
            deps.validation_errors = routing_decision.validation_errors

            logger.info(
                f"Function detector found {len(routing_decision.detected_functions)} function(s): "
                f"{[f['name'] for f in routing_decision.detected_functions]}",
            )

            if routing_decision.validation_errors:
                logger.warning(
                    f"Validation errors detected: {routing_decision.validation_errors}"
                )

        # Log the detection but don't restrict Gemini's choices
        logger.debug("Function detection complete. Gemini will decide search strategy.")

    except ImportError as e:
        logger.warning(f"Function detector not available: {e}")
        # Continue without function detection
    except Exception as e:
        logger.error(f"Function detection failed: {e}")
        # Continue without function detection - fail gracefully


async def run_pulsepal_stream(
    query: str,
    session_id: str = None,
    temperature: float = 0.1,
):
    """
    Run a query with streaming response.

    Args:
        query: User query
        session_id: Optional session ID
        temperature: LLM temperature

    Yields:
        Response chunks
    """
    try:
        # Create or get session with semantic routing
        session_id, deps = await create_pulsepal_session(session_id, query)

        # Add query to conversation history
        deps.conversation_context.add_conversation("user", query)

        # Run the agent with streaming
        async with pulsepal_agent.run_stream(
            query,
            deps=deps,
            model_settings={"temperature": temperature},
        ) as result:
            full_response = ""
            async for chunk in result.stream():
                full_response += chunk
                yield chunk

        # Add complete response to conversation history
        deps.conversation_context.add_conversation("assistant", full_response)

    except GeminiRecitationError as e:
        logger.warning(f"Recitation error during streaming: {e}")
        yield (
            "\n\nI encountered an issue with content generation. "
            "Please try rephrasing your question."
        )
    except Exception as e:
        # Log the full error with traceback for debugging
        import traceback

        logger.error(f"Error during streaming: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")

        # Check for specific error types
        error_msg = str(e).lower()

        if "503" in error_msg or "service unavailable" in error_msg:
            logger.error("Gemini API returned 503 during streaming")
            yield (
                "\n\n⚠️ **Gemini API Service Unavailable (503)**\n"
                "The AI service is temporarily unavailable. Please try again in a few moments.\n"
                f"Error: {e!s}"
            )
        elif "rate limit" in error_msg or "429" in error_msg:
            logger.error("Rate limit exceeded during streaming")
            yield (
                "\n\n⚠️ **Rate Limit Exceeded**\n"
                "Please wait a moment before trying again.\n"
                f"Error: {e!s}"
            )
        elif "timeout" in error_msg:
            logger.error("Timeout during streaming")
            yield (
                "\n\n⚠️ **Request Timeout**\n"
                "The request took too long. Try a simpler query.\n"
                f"Error: {e!s}"
            )
        else:
            # For unexpected errors, provide the actual error
            logger.error(f"Unexpected streaming error: {type(e).__name__}")
            yield (
                f"\n\n❌ **Error: {type(e).__name__}**\n"
                f"An unexpected error occurred: {e!s}\n"
                "This error has been logged for debugging."
            )
