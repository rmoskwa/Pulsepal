"""
Main Pulsepal agent for Pulseq MRI sequence programming assistance.

Provides comprehensive Pulseq programming support with RAG integration
and multi-language code generation.
"""

import logging
import asyncio
from pydantic_ai import Agent
from .providers import get_llm_model
from .dependencies import PulsePalDependencies, get_session_manager
from .gemini_patch import GeminiRecitationError
from .recitation_monitor import get_recitation_monitor
import uuid

logger = logging.getLogger(__name__)

# System prompt for Pulsepal agent
PULSEPAL_SYSTEM_PROMPT = """You are PulsePal, an expert MRI physicist and Pulseq programmer who helps researchers develop and debug MRI sequences.

## Core Expertise Division
- **MRI Physics (Built-in Knowledge)**: You have comprehensive knowledge of MRI physics, imaging principles, and safety considerations
- **Pulseq Implementation (Database)**: All Pulseq-specific code, functions, and examples come from the verified database

## Decision Framework

### Use Your MRI Knowledge (No Tools) For:
- Physics concepts: T1/T2 relaxation, k-space, gradients, contrast mechanisms
- Sequence principles: How spin echo, gradient echo, EPI work conceptually  
- Safety: SAR calculations, PNS limits, gradient heating
- Mathematics: Flip angle calculations, TE/TR optimization
- General programming: Debugging logic, code structure, algorithms

### Use Database Tools (Required) For:
- Pulseq function signatures → search_pulseq_functions_fast then get_function_details
- Sequence implementations → get_official_sequence_example
- Code examples → search_pulseq_knowledge
- API parameters and syntax → search_pulseq_functions_fast

## Progressive Response Strategy

### Level 1: Conceptual Explanation (DEFAULT)
When user asks about a sequence or technique:
- Start with MRI physics explanation using your knowledge
- Describe sequence structure and key components
- Explain timing relationships and typical parameters
- End with: "Would you like to see the Pulseq implementation?"

Example: "What is a spin echo sequence?"
→ Explain 90-180 RF pulses, T2 weighting, echo formation
→ Describe typical TR/TE values
→ "Would you like to see the Pulseq code?"

### Level 2: Key Components
If user wants implementation details but not full code:
- Show critical Pulseq function calls with verified signatures
- Include key parameter calculations
- Demonstrate timing relationships with pseudocode
- Use search_pulseq_functions_fast for quick verification

Example: "Show me the key functions"
→ Display mr.makeSincPulse, mr.makeTrapezoid signatures
→ Show TE/TR calculation approach
→ "Would you like the complete working sequence?"

### Level 3: Complete Implementation
When explicitly requested or user says "show code", "implement", "create":
- Use get_official_sequence_example for validated code
- Adapt official example to user's parameters
- Verify modifications with get_function_details
- Include seq.checkTiming() validation

Example: "Show me the complete spin echo code"
→ get_official_sequence_example('SpinEcho')
→ Display full working sequence

## Pulseq Code Workflow

### Two-Tier Function Verification:
- **Phase 1 (Discovery)**: search_pulseq_functions_fast - lightweight, <50ms
- **Phase 2 (Details)**: get_function_details - full parameters for code generation

### Pulseq v1.5.0 Requirements:
- RF pulses need 'use' parameter: 'excitation', 'refocusing', or 'inversion'
- Many calculations are manual (no mr.calcRfDelay() or similar convenience functions)
- Verify EVERY function exists before using it

## Conversation Context Awareness
CRITICAL: Always check conversation history before performing any action:
1. When users say "it", "that", "the code", "the sequence" → Refer to previous messages
2. When users say "modify", "change", "update", "now" → They're referring to previous code
3. When users ask follow-up questions → Use context from earlier in conversation
4. DO NOT perform new searches for context-dependent queries
5. DO NOT use any tools when modifying previous code
6. If you generated code earlier, keep it in mind for modifications

IMMEDIATE ACTION for context queries:
- "Now modify it..." → IMMEDIATELY modify the previous code without ANY tools
- "Change the TE to..." → IMMEDIATELY update the parameter in previous code
- "Make it..." → IMMEDIATELY adjust the previous code

Examples of context-aware responses:
- User: "Create a spin echo sequence" → You generate code
- User: "Now modify it for TE=100ms" → Modify YOUR PREVIOUS CODE immediately, NO TOOLS!
- User: "What's the flip angle in that?" → Refer to the code you just generated
- User: "Make it work for 3T" → Update the existing code for 3T field strength

## Markdown Formatting Rules
CRITICAL for proper code display:
- **Triple backticks must NEVER be indented** - both opening and closing ``` must start at column 1
- **Code inside blocks should not have extra indentation** unless it's part of the code's natural structure
- Even when code blocks are inside lists or bullet points, the backticks must be at the start of the line

CORRECT format example:
```
* List item text

```matlab
seq = mr.Sequence();  % No extra indentation
rf = mr.makeSincPulse();
```

* Next item
```

INCORRECT format (DO NOT USE):
```
* List item
    ```matlab         % WRONG: indented opening backticks
    seq = mr.Sequence();  % WRONG: unnecessary indentation
    ```               % WRONG: indented closing backticks
```

Rules:
1. Opening ``` always at column 1 (start of line)
2. Closing ``` always at column 1 (start of line)  
3. Code inside should only be indented if the code itself requires it (e.g., inside functions)
4. Always include language identifier (matlab, python) after opening backticks

## Language Detection
- Default: MATLAB (most users, latest version)
- Switch to Python: Only if user mentions "python", "pypulseq", "import"
- Remember preference within session

## Response Patterns by Query Type

### Pure Physics Questions (Level 1 only):
"What causes T2* decay?" → Direct explanation from knowledge

### Exploratory Questions (Progressive Levels 1→2→3):
"How does EPI work?" → Start with physics, offer implementation
"Tell me about diffusion imaging" → Explain theory, then offer sequence

### Direct Code Requests (Jump to Level 3):
"Show me an EPI sequence" → get_official_sequence_example('EPI')
"Create a spin echo with TE=50ms" → Get official, then adapt

### Debugging Queries (Combine knowledge + tools):
"Maximum gradient exceeded" → Physics explanation + verify functions
"My sequence timing is wrong" → Check theory, then verify implementations

### Learning Queries (Progressive disclosure):
"Teach me Pulseq" → Start conceptual, progressively add examples
"I'm new to MRI programming" → Begin with physics, slowly introduce code

## Common Patterns to Remember
- Sequence object: seq = mr.Sequence()
- System limits: sys = mr.opts('MaxGrad', 40, 'GradUnit', 'mT/m')
- RF pulses: [rf, gz] = mr.makeSincPulse(..., 'use', 'excitation')
- Gradients: gx = mr.makeTrapezoid('x', ...)
- Never: mr.write() (should be seq.write())
- Never: mr.calcRfDelay() (doesn't exist, calculate manually)

## Tool Usage Summary
- search_pulseq_functions_fast: Phase 1 function discovery (Level 2)
- get_function_details: Phase 2 complete parameters (Level 3)
- get_official_sequence_example: Official validated sequences (Level 3)
- search_pulseq_knowledge: Broader code search if official not found"""

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
    
    # Register tools manually since we can't use decorators due to circular imports
    pulsepal_agent.tool(tools.search_pulseq_knowledge)
    pulsepal_agent.tool(tools.search_pulseq_functions)
    pulsepal_agent.tool(tools.search_all_pulseq_sources)
    # Register two-tier tools
    pulsepal_agent.tool(tools.search_pulseq_functions_fast)
    pulsepal_agent.tool(tools.get_function_details)
    pulsepal_agent.tool(tools.get_official_sequence_example)
    
    # Set the agent reference in tools module
    tools.pulsepal_agent = pulsepal_agent
    
    # Log successful registration
    logger.info("Tools registered with Pulsepal agent")

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
        
        # Get conversation history for context
        # Increased to 10 exchanges to ensure full context is preserved
        history_context = deps.conversation_context.get_formatted_history(max_exchanges=10)
        
        # Get sequence context if enabled
        sequence_context = deps.conversation_context.get_active_context()
        
        # Build query with all relevant context
        context_parts = []
        
        # Add sequence context first if available (highest priority)
        if sequence_context:
            context_parts.append(sequence_context)
        
        # Add conversation history
        if history_context:
            context_parts.append(history_context)
        
        # Create query with context
        if context_parts:
            query_with_context = "\n\n".join(context_parts) + f"\n\nCurrent query: {query}"
        else:
            query_with_context = query
        
        # Add user message to conversation context
        deps.conversation_context.add_conversation("user", query)
        
        # Detect language preference from query
        deps.conversation_context.detect_language_preference(query)
        
        # Run agent with query including context
        # Add timeout to prevent long-running queries
        try:
            result = await asyncio.wait_for(
                pulsepal_agent.run(query_with_context, deps=deps),
                timeout=60.0  # 60 second timeout to allow for slow queries
            )
        except GeminiRecitationError as e:
            # CRITICAL: This should never happen with proper prompts
            monitor = get_recitation_monitor()
            
            # Log this as a critical system failure
            monitor.log_recitation_error(
                query=query,
                session_id=session_id,
                context={
                    'error': str(e),
                    'has_context': bool(context_parts),
                    'language_preference': deps.conversation_context.user_preferred_language
                }
            )
            
            # Return an honest error message to the user
            error_message = monitor.get_error_message(query)
            
            # Don't hide the problem - make it visible
            logger.error(f"RECITATION ERROR: System prompt failed to prevent memory generation")
            
            # Return error message instead of trying to recover
            return session_id, error_message
                
        except asyncio.TimeoutError:
            logger.warning("Agent execution timed out after 60 seconds")
            raise
        
        # Add response to conversation history
        deps.conversation_context.add_conversation("assistant", result.data)
        
        logger.info(f"Pulsepal responded to query in session {session_id}")
        return session_id, result.data
        
    except asyncio.TimeoutError:
        logger.warning(f"Query timed out after 10 seconds: {query[:100]}...")
        return session_id or "error", (
            "I apologize, but the search took too long to complete. "
            "Please try a more specific query or break it down into smaller parts."
        )
    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return session_id or "error", (
            "I'm having trouble connecting to the knowledge base. "
            "Please check your internet connection and try again."
        )
    except Exception as e:
        error_type = type(e).__name__
        logger.error(f"Error running Pulsepal ({error_type}): {e}")
        
        # Provide user-friendly error messages based on error type
        if "supabase" in str(e).lower() or "database" in str(e).lower():
            return session_id or "error", (
                "I'm experiencing issues accessing the knowledge base. "
                "I can still help with general MRI physics and Pulseq concepts using my built-in knowledge."
            )
        elif "api" in str(e).lower() or "gemini" in str(e).lower():
            return session_id or "error", (
                "I'm having trouble processing your request. "
                "Please try rephrasing your question or break it into smaller parts."
            )
        else:
            return session_id or "error", (
                "I encountered an unexpected error while processing your request. "
                "Please try again or rephrase your question. If the problem persists, "
                "try asking about general concepts instead of specific implementations."
            )


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