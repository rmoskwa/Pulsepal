"""
Main Pulsepal agent with modern RAG service v2.

Simplified version that lets the LLM handle all intelligence,
while RAG just retrieves documents.
"""

import logging
from pydantic_ai import Agent
from .providers import get_llm_model
from .dependencies import PulsePalDependencies, get_session_manager
from .gemini_patch import GeminiRecitationError
import uuid

logger = logging.getLogger(__name__)

# Improved system prompt with better information disclosure strategy
PULSEPAL_SYSTEM_PROMPT = """You are PulsePal, an expert MRI physicist and Pulseq programming assistant.

## Core Expertise
You possess deep understanding of:
- MRI physics and pulse sequence design principles
- Pulseq framework for both MATLAB and Python (pypulseq)
- Common implementation patterns and debugging strategies

## Information Retrieval System

You have access to three specialized knowledge sources:

1. **api_reference**: Complete function documentation with ALL parameters and returns
2. **crawled_pages**: Implementation examples and tutorials
3. **official_sequence_examples**: Validated educational sequences

### Direct Function Lookup
When specific functions are mentioned (e.g., "calculateKspacePP"), the system may perform direct database lookup, providing comprehensive documentation. Trust this information - it's authoritative.

## CRITICAL: Information Disclosure Strategy

### Level 1 - Initial Response
For general questions, provide commonly-used information (80% use case):
- Core parameters and returns
- Essential functionality
- Mention if additional options exist: "Additional parameters/returns are available for advanced use."

### Level 2 - Follow-Up or Specific Questions
When users:
- Ask "are there more parameters/returns?"
- Question your response ("I thought...")
- Ask about specific features ("what about phase modulation?")
- Express confusion or doubt

IMMEDIATELY provide COMPLETE information from the database:
- ALL parameters (required and optional)
- ALL return values
- Full function signatures
- Don't hide "advanced" features

### Level 3 - Explicit Completeness
For requests containing "full", "all", "complete", "documentation", "every":
- Provide exhaustive information
- Include all optional parameters with descriptions
- List all return values with explanations
- Show complete usage examples

## Trust the Database
When api_reference contains comprehensive documentation:
- Present it accurately and completely when asked
- Don't omit parameters marked as optional
- If the database lists 10 returns, mention all 10 (not just 6)
- The database is more authoritative than common usage patterns

## Query Response Guidelines

### IMPORTANT: Check for Validation Errors
If deps.validation_errors contains namespace issues:
1. **Immediately inform the user** of the namespace error(s)
2. **Suggest the correct form** (e.g., "seq.makeAdc should be mr.makeAdc")
3. **Then search using the CORRECT form**, not the user's incorrect form
4. **Never fabricate documentation** for incorrectly namespaced functions

Example: If user asks about "seq.makeAdc" but validation says it should be "mr.makeAdc":
- Tell user: "Note: seq.makeAdc is not valid. The correct form is mr.makeAdc"
- Then search for and provide documentation about mr.makeAdc

### Standard Guidelines
1. **Function Validation**: MUST validate every function name mentioned
2. **Namespace Verification**: Strictly enforce correct usage (mr.* vs seq.*)
3. **Progressive Complexity**: Start simple, but provide complete info when asked
4. **Session Awareness**: Use conversation history to understand user expertise level

## Language Handling
- Default to MATLAB unless Python is specified
- When showing function signatures, use the language context
- For general concepts, mention both if relevant

## Error Recovery
When users challenge your response:
1. Re-check the database for complete information
2. Provide the full documentation
3. Acknowledge if initial response was incomplete
4. Don't defend simplification if user needs more

## Search Strategy Selection

Use search when:
- User asks for examples or implementations
- Debugging specific issues
- Learning sequence construction
- Comparing approaches

Skip search when:
- Answering pure physics questions
- User provides complete code for review
- Discussing general MRI concepts

Remember: Users who ask follow-up questions about parameters usually need complete information, not continued simplification."""

# Create Pulsepal agent
pulsepal_agent = Agent(
    get_llm_model(),
    deps_type=PulsePalDependencies,
    system_prompt=PULSEPAL_SYSTEM_PROMPT,
)


# Import and register tools after agent creation
def _register_tools():
    """Register enhanced tools with the pulsepal agent."""
    from . import tools_v2 as tools

    # Register modern tools with validation
    pulsepal_agent.tool(tools.search_pulseq_knowledge)
    pulsepal_agent.tool(tools.verify_function_namespace)
    pulsepal_agent.tool(
        tools.validate_pulseq_function
    )  # Critical for hallucination prevention
    pulsepal_agent.tool(tools.validate_code_block)  # Validate entire code blocks

    # Set the agent reference in tools module
    tools.pulsepal_agent = pulsepal_agent

    logger.info("Modern tools registered with Pulsepal agent")


# Register tools on module import
_register_tools()


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

    # Apply semantic routing if query provided
    if query:
        try:
            from .semantic_router import SemanticRouter, QueryRoute

            router = SemanticRouter()
            routing_decision = router.classify_query(query)

            # Set routing flags based on decision
            if routing_decision.route == QueryRoute.FORCE_RAG:
                deps.force_rag = True
                deps.forced_search_hints = routing_decision.search_hints
                deps.detected_functions = routing_decision.detected_functions  # Pass detected functions
                deps.validation_errors = routing_decision.validation_errors  # Pass validation errors
                
                if routing_decision.detected_functions:
                    logger.info(
                        f"Semantic router: FORCE_RAG - {routing_decision.reasoning} - "
                        f"Detected functions: {[f['name'] for f in routing_decision.detected_functions]}"
                    )
                    if routing_decision.validation_errors:
                        logger.warning(f"Validation errors: {routing_decision.validation_errors}")
                else:
                    logger.info(
                        f"Semantic router: FORCE_RAG - {routing_decision.reasoning}"
                    )
            elif routing_decision.route == QueryRoute.NO_RAG:
                deps.skip_rag = True
                logger.info(f"Semantic router: NO_RAG - {routing_decision.reasoning}")
            else:
                logger.info(
                    f"Semantic router: GEMINI_CHOICE - {routing_decision.reasoning}"
                )
        except ImportError as e:
            logger.warning(f"Semantic router not available: {e}")
            # Continue without routing - let LLM decide
        except Exception as e:
            logger.error(f"Semantic routing failed: {e}")
            # Continue without routing - fail gracefully
            # This ensures the application doesn't crash if routing fails

    logger.info(f"Created Pulsepal session: {session_id}")

    return session_id, deps


async def run_pulsepal_query(
    query: str, session_id: str = None, temperature: float = 0.1
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
            query, deps=deps, model_settings={"temperature": temperature}
        )

        # Extract response
        response = result.data if hasattr(result, "data") else str(result)

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
        logger.error(f"Error running Pulsepal query: {e}")
        return session_id, f"An error occurred: {str(e)}"


def apply_semantic_routing(query: str, deps: PulsePalDependencies) -> None:
    """
    Apply semantic routing to a query and update dependencies.
    This is a standalone function for use by Chainlit.

    Args:
        query: The user query to route
        deps: Dependencies object to update with routing decisions
    """
    try:
        from .semantic_router import SemanticRouter, QueryRoute

        router = SemanticRouter()
        routing_decision = router.classify_query(query)

        # Set routing flags based on decision
        if routing_decision.route == QueryRoute.FORCE_RAG:
            deps.force_rag = True
            deps.forced_search_hints = routing_decision.search_hints
            logger.info(f"Semantic router: FORCE_RAG - {routing_decision.reasoning}")
        elif routing_decision.route == QueryRoute.NO_RAG:
            deps.skip_rag = True
            logger.info(f"Semantic router: NO_RAG - {routing_decision.reasoning}")
        else:
            logger.info(
                f"Semantic router: GEMINI_CHOICE - {routing_decision.reasoning}"
            )
    except ImportError as e:
        logger.warning(f"Semantic router not available: {e}")
        # Continue without routing - let LLM decide
    except Exception as e:
        logger.error(f"Semantic routing failed: {e}")
        # Continue without routing - fail gracefully


async def run_pulsepal_stream(
    query: str, session_id: str = None, temperature: float = 0.1
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
            query, deps=deps, model_settings={"temperature": temperature}
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
        logger.error(f"Error during streaming: {e}")
        yield f"\n\nAn error occurred: {str(e)}"
