"""
Main Pulsepal agent for Pulseq MRI sequence programming assistance.

Provides comprehensive Pulseq programming support with RAG integration,
agent delegation to MRI Expert, and multi-language code generation.
"""

import logging
from pydantic_ai import Agent, RunContext
from .providers import get_llm_model
from .dependencies import PulsePalDependencies, get_session_manager
import uuid

logger = logging.getLogger(__name__)

# System prompt for Pulsepal agent
PULSEPAL_SYSTEM_PROMPT = """You are Pulsepal, an expert AI assistant for Pulseq MRI sequence programming. 

You help researchers and programmers work with Pulseq v1.5.0 across MATLAB, Octave, and Python environments.

## IMPORTANT: Tool Usage

YOU MUST USE YOUR TOOLS TO SEARCH FOR INFORMATION. When users ask questions:
1. ALWAYS use `perform_rag_query` to search documentation FIRST
2. Use `search_code_examples` to find specific code implementations
3. Use `get_available_sources` to discover what documentation is available
4. Use `delegate_to_mri_expert` for physics explanations

DO NOT provide generic responses. ALWAYS search your knowledge base using the tools provided.

## Your Core Capabilities:

1. **Code Generation**: Generate Pulseq sequences in MATLAB, Octave, and Python
2. **Debugging**: Help fix sequence errors and optimization issues  
3. **Language Conversion**: Convert between MATLAB/Octave and Python implementations
4. **Documentation**: Access comprehensive Pulseq documentation via RAG tools
5. **Physics Consultation**: Delegate complex MRI physics questions to the MRI Expert

## Your Tools (USE THESE ACTIVELY):

- `perform_rag_query`: Search Pulseq documentation and examples - USE THIS FOR ALL DOCUMENTATION QUESTIONS
- `search_code_examples`: Find specific code examples and implementations - USE THIS FOR CODE QUESTIONS
- `get_available_sources`: Discover available documentation sources - USE THIS TO EXPLORE AVAILABLE CONTENT
- `delegate_to_mri_expert`: Get expert physics explanations and educational content - USE THIS FOR PHYSICS QUESTIONS

## Programming Languages You Support:

- **MATLAB**: Primary Pulseq environment with .seq file generation
- **Octave**: Open-source alternative to MATLAB
- **Python**: Using pulseq-python package for sequence programming

## Response Guidelines:

- FIRST: Use tools to search for relevant information
- THEN: Provide answers based on the search results
- Always check user's preferred language or detect from context
- Provide working, tested code examples when possible
- Include clear comments explaining sequence logic
- Reference specific Pulseq functions and parameters
- Warn about potential scanner safety issues

## When to Delegate:

Delegate to MRI Expert for:
- Fundamental MRI physics explanations
- k-space trajectory analysis
- RF pulse design theory
- Gradient timing calculations
- Scanner hardware limitations
- Educational content about MRI principles

Keep programming questions for yourself:
- Pulseq syntax and API usage
- Code debugging and optimization
- Language-specific implementation details
- Sequence file generation and validation

Always maintain conversation context and build upon previous code examples when relevant."""

# Create Pulsepal agent
pulsepal_agent = Agent(
    get_llm_model(),
    deps_type=PulsePalDependencies,
    system_prompt=PULSEPAL_SYSTEM_PROMPT,
)

# Import and register tools after agent creation to avoid circular imports
def _register_tools():
    """Register tools with the pulsepal agent."""
    from . import tools
    # Tools are registered via decorators when the module is imported

# Register tools on module import
_register_tools()


async def create_pulsepal_session(session_id: str = None) -> tuple[str, PulsePalDependencies]:
    """
    Create a new Pulsepal session with initialized dependencies.
    
    Args:
        session_id: Optional session ID, generates UUID if not provided
        
    Returns:
        tuple: (session_id, initialized_dependencies)
    """
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    # Get session manager and create/get session context
    session_manager = get_session_manager()
    conversation_context = session_manager.get_session(session_id)
    
    # Create dependencies
    deps = PulsePalDependencies(
        conversation_context=conversation_context,
        session_manager=session_manager
    )
    
    # Initialize MCP server connection
    await deps.initialize_mcp_server()
    
    logger.info(f"Created Pulsepal session: {session_id}")
    return session_id, deps


async def run_pulsepal(query: str, session_id: str = None) -> tuple[str, str]:
    """
    Run Pulsepal agent with a query.
    
    Args:
        query: User query for Pulseq assistance
        session_id: Optional session ID for conversation continuity
        
    Returns:
        tuple: (session_id, agent_response)
    """
    try:
        # Create or get session
        if session_id is None:
            session_id, deps = await create_pulsepal_session()
        else:
            session_manager = get_session_manager() 
            conversation_context = session_manager.get_session(session_id)
            deps = PulsePalDependencies(
                conversation_context=conversation_context,
                session_manager=session_manager
            )
            await deps.initialize_mcp_server()
        
        # Add user query to conversation history
        deps.conversation_context.add_conversation("user", query)
        
        # Detect language preference if not already set
        if not deps.conversation_context.preferred_language:
            deps.conversation_context.detect_language_preference(query)
        
        # Run agent
        result = await pulsepal_agent.run(query, deps=deps)
        
        # Add assistant response to conversation history
        deps.conversation_context.add_conversation("assistant", result.data)
        
        logger.info(f"Pulsepal responded to query in session {session_id}")
        return session_id, result.data
        
    except Exception as e:
        error_msg = f"Error running Pulsepal: {e}"
        logger.error(error_msg)
        return session_id or "error", f"I apologize, but I encountered an error: {error_msg}. Please try again."


# For backward compatibility and simple usage
async def ask_pulsepal(query: str) -> str:
    """
    Simple interface to ask Pulsepal a question without session management.
    
    Args:
        query: User query for Pulseq assistance
        
    Returns:
        str: Agent response
    """
    session_id, response = await run_pulsepal(query)
    return response