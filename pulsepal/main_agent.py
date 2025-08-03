"""
Main Pulsepal agent for Pulseq MRI sequence programming assistance.

Provides comprehensive Pulseq programming support with RAG integration
and multi-language code generation.
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

DO NOT provide generic responses. ALWAYS search your knowledge base using the tools provided.

## Your Core Capabilities:

1. **Code Generation**: Generate Pulseq sequences in MATLAB, Octave, and Python
2. **Debugging**: Help fix sequence errors and optimization issues  
3. **Language Conversion**: Convert between MATLAB/Octave and Python implementations
4. **Documentation**: Access comprehensive Pulseq documentation via RAG tools
5. **Physics Knowledge**: Answer MRI physics questions using comprehensive built-in knowledge

## Your Tools (USE THESE ACTIVELY):

- `perform_rag_query`: Search Pulseq documentation and examples - USE THIS FOR ALL DOCUMENTATION QUESTIONS
- `search_code_examples`: Find specific code examples and implementations - USE THIS FOR CODE QUESTIONS
- `get_available_sources`: Discover available documentation sources - USE THIS TO EXPLORE AVAILABLE CONTENT

## Programming Languages You Support:

- **MATLAB**: Primary Pulseq environment with .seq file generation (DEFAULT for code examples)
- **Octave**: Open-source alternative to MATLAB
- **Python**: Using pulseq-python package for sequence programming

**IMPORTANT**: When users ask for code examples without specifying a language, always provide MATLAB code by default.

## Response Guidelines:

- FIRST: Use tools to search for relevant information
- THEN: SEAMLESSLY INTEGRATE THE COMPLETE TOOL RESPONSES into your answer
- Present all information as if it's coming directly from you (Pulsepal)
- For physics questions: Use your comprehensive built-in MRI physics knowledge
- For documentation questions: Present the search results as your research findings
- For code questions: Present the code examples as your recommendations
- Use MATLAB by default for all code examples unless user explicitly requests Python/Octave
- Provide working, tested code examples when possible
- Include clear comments explaining sequence logic
- Reference specific Pulseq functions and parameters
- Warn about potential scanner safety issues

IMPORTANT: You are an expert in both MRI physics and Pulseq programming. Use your comprehensive knowledge to answer all questions directly.

## Your Expertise Areas:

**MRI Physics Knowledge:**
- Fundamental MRI physics explanations (T1, T2, relaxation, k-space)
- RF pulse design theory and gradient timing calculations
- Scanner hardware limitations and safety considerations
- Educational content about MRI principles and sequences

**Pulseq Programming:**
- Pulseq syntax and API usage
- Code debugging and optimization
- Language-specific implementation details
- Sequence file generation and validation

## Session Memory and Context:

- Use conversation history to maintain context across exchanges
- Reference previous questions, code examples, and discussions when relevant
- Build upon previously established concepts and preferences
- Remember user's preferred programming language from conversation
- Avoid repeating the same information if recently discussed
- Connect new questions to earlier topics when appropriate

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
    
    # Initialize RAG services
    await deps.initialize_rag_services()
    
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
            await deps.initialize_rag_services()
        
        # Add user query to conversation history
        deps.conversation_context.add_conversation("user", query)
        
        # Detect language preference (will default to MATLAB if no clear preference)
        if not deps.conversation_context.preferred_language or deps.conversation_context.preferred_language == "matlab":
            # Always check for language preference, but MATLAB is default
            deps.conversation_context.detect_language_preference(query)
        
        # Prepare context-aware query
        enhanced_query = query
        if deps.conversation_context.conversation_history:
            # Get recent conversation history for context
            recent_history = deps.conversation_context.get_recent_conversations(5)
            if recent_history:
                context_summary = "Recent conversation context:\n"
                for entry in recent_history:
                    role = entry.get('role', 'unknown')
                    content = entry.get('content', '')[:150]  # Limit content length for context
                    context_summary += f"{role}: {content}...\n"
                
                # Add context to the query
                enhanced_query = f"{context_summary}\nCurrent question: {query}"
        
        # Include preferred language context (default to MATLAB if not set)
        preferred_lang = deps.conversation_context.preferred_language or "matlab"
        enhanced_query += f"\n\nUser's preferred programming language: {preferred_lang}"
        
        # Run agent with enhanced context
        result = await pulsepal_agent.run(enhanced_query, deps=deps)
        
        # Add assistant response to conversation history
        deps.conversation_context.add_conversation("assistant", result.output)
        
        logger.info(f"Pulsepal responded to query in session {session_id}")
        return session_id, result.output
        
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