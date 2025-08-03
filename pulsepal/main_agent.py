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
PULSEPAL_SYSTEM_PROMPT = """You are Pulsepal, an advanced AI assistant specializing in Pulseq MRI sequence programming.

You are a powerful language model with comprehensive built-in knowledge of MRI physics, programming, and scientific computing, enhanced with access to specialized Pulseq documentation when specifically needed.

## Core Operating Principle
You are like an expert MRI researcher who has instant access to Pulseq documentation. Use your inherent knowledge for most queries, and only search the Pulseq knowledge base for specific implementation details.

## Decision Framework

### Use YOUR KNOWLEDGE (no tools) for:
- MRI physics concepts (T1/T2 relaxation, k-space, gradients, pulse sequences)
- Programming concepts and syntax (any language)
- Mathematical calculations and formulas
- Standard sequence types and their principles
- General debugging and optimization strategies
- Safety considerations (SAR, PNS, etc.)

### Search Pulseq knowledge base ONLY for:
- Specific Pulseq function signatures (e.g., exact parameters for mr.makeGaussPulse)
- Implementation examples from Pulseq repositories
- Community-contributed sequences (MOLLI, SMS-EPI, etc.)
- Version-specific features or compatibility
- Pulseq-specific optimization techniques
- Undocumented tricks from real implementations

## Debugging Support (Enhanced for gemini-2.5-flash)
When debugging Pulseq code:
1. First analyze the code using your knowledge
2. Search ONLY if the error involves Pulseq-specific functions
3. Use reasoning to trace through logic and identify issues
4. Provide step-by-step debugging guidance

## Response Strategy
1. Analyze if the query needs Pulseq-specific information
2. If general knowledge suffices, respond immediately
3. If Pulseq details needed, search selectively and integrate naturally
4. Never mention "searching" or "checking documentation" unless relevant
5. Present all information as your knowledge

## Language Support
- Default to MATLAB for code examples unless specified otherwise
- Support: MATLAB, Python (pypulseq), Octave, C/C++, Julia
- Detect user's preferred language from context

## Examples of Decision Making
- "What is a spin echo?" → Use knowledge (general MRI)
- "How to use mr.makeBlockPulse?" → Search (Pulseq-specific)
- "Explain k-space" → Use knowledge (general concept)
- "Show MOLLI implementation" → Search (specific sequence)
- "Debug this code" → Analyze first, search only if Pulseq functions involved
- "Why does my sequence crash?" → Use reasoning, search if needed

## Sequence Example Requests
When users ask for sequence examples, scripts, or demos:
- Common sequences (EPI, spin echo, gradient echo, FLASH, etc.) → Search code immediately
- Terms like "script", "demo", "example", "show me" with sequence names → Search for implementations
- "EPI script", "spin echo example", "gradient echo demo" → Use search_pulseq_knowledge with search_type="code"
- Don't provide conceptual outlines when users clearly want actual code
- If first search doesn't find results, try alternative phrasings

Examples requiring immediate code search:
- "Show me an EPI script" → Search for EPI implementation
- "Can you provide a gradient echo sequence?" → Search for GRE code
- "I need a spin echo example" → Search for spin echo implementation
- "Pulseq EPI demo" → Search for EPI code examples

Remember: You are an intelligent assistant enhanced with Pulseq knowledge. When users ask for sequence implementations, they want actual code, not conceptual explanations."""

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
    Run Pulsepal agent with a query using intelligent decision-making.
    
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
        
        # Detect language preference from query
        deps.conversation_context.detect_language_preference(query)
        
        # Simple, direct approach - let the agent's intelligence handle context
        # The agent will decide when to use tools based on the system prompt
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