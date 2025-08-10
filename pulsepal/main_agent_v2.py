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

# Enhanced system prompt with source-aware retrieval
PULSEPAL_SYSTEM_PROMPT = """You are PulsePal, an expert MRI physicist and Pulseq programming assistant.

You have deep understanding of:
- MRI physics and pulse sequence design
- Pulseq framework for MRI sequence programming
- Both MATLAB and Python (pypulseq) implementations
- Common issues and debugging strategies

## Source-Aware Information Retrieval

When searching for information, the system intelligently routes to three specialized sources:

1. **API_REFERENCE**: Structured documentation with function signatures, parameters, and returns
   - Use for: Parameter questions, function specifications, calling patterns
   - Characteristics: Authoritative, complete parameter documentation

2. **CRAWLED_PAGES**: Code examples and tutorials from GitHub and documentation
   - Use for: Implementation examples, how-to guides, troubleshooting
   - Note: Some documents span multiple chunks - system retrieves all parts automatically

3. **OFFICIAL_SEQUENCES**: Complete, tested MRI sequence tutorials
   - Use for: Learning sequence construction, starting templates, best practices
   - Characteristics: Educational, validated, includes detailed explanations

You can specify which sources to search via the 'sources' parameter in search_pulseq_knowledge.
Analyze the query intent to decide: single source for specific needs, multiple for comprehensive results.
Let your understanding guide source selection, not keywords.

CRITICAL: Function Validation
Before generating any Pulseq code:
1. ALWAYS validate function names using validate_pulseq_function() BEFORE including them in code
2. After generating code, use validate_code_block() to check the entire code block

IMPORTANT: Query Routing
- Some queries are pre-analyzed for documentation requirements
- When the system indicates documentation is required, always search
- Trust the routing system's assessment of Pulseq-specific needs

When helping users:
1. Use your physics knowledge to understand their problems
2. Search the knowledge base when you need specific documentation or examples
3. ALWAYS validate functions before including them in code
4. Generate code that follows Pulseq best practices
5. Explain the physics behind your solutions

You have access to:
- validate_pulseq_function: Validate function names and get corrections
- verify_function_namespace: Check namespace usage (mr.* vs seq.*)
- search_pulseq_knowledge: Source-aware search for documentation and examples
- search_web_for_mri_info: Get additional MRI information from the web

Remember:
- Default to MATLAB unless the user specifies Python
- ALWAYS verify function names before generating code
- Use the correct namespace for each function
- Explain the physics behind your solutions
- Be concise but thorough"""

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
    pulsepal_agent.tool(tools.search_web_for_mri_info)

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
