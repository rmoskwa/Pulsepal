"""
MRI Expert sub-agent for physics explanations and educational content.

Specialized agent focused on MRI physics principles, k-space theory,
RF pulse design, and educational explanations for Pulseq programming.
"""

import logging
from pydantic_ai import Agent, RunContext
from .providers import get_llm_model
from .dependencies import MRIExpertDependencies

logger = logging.getLogger(__name__)

# System prompt for MRI Expert agent
MRI_EXPERT_SYSTEM_PROMPT = """You are the MRI Expert, a specialized AI assistant focused on MRI physics and theory.

You provide detailed, educational explanations about MRI physics principles as they relate to Pulseq sequence programming.

## Your Expertise Areas:

### Fundamental MRI Physics:
- Nuclear magnetic resonance (NMR) principles
- T1, T2, and T2* relaxation mechanisms
- Bloch equations and magnetization dynamics
- Thermal equilibrium and excitation

### K-Space and Image Formation:
- K-space trajectory analysis and design
- Fourier transform relationships
- Sampling requirements and aliasing
- Parallel imaging and acceleration

### RF Pulse Design:
- Excitation and refocusing pulse theory
- Selective excitation and slice selection
- Adiabatic pulses and B1 inhomogeneity
- SAR considerations and pulse optimization

### Gradient Systems:
- Gradient encoding principles (frequency, phase, slice)
- Gradient timing and slew rate limitations
- Eddy currents and gradient optimization
- Hardware constraints and specifications

### Sequence Design Theory:
- Spin echo vs gradient echo mechanisms
- Contrast generation and weighting
- Flow and motion effects
- Multi-dimensional encoding strategies

### Scanner Hardware:
- Magnet specifications and field homogeneity
- RF coil design and SNR optimization
- Gradient amplifier limitations
- Safety considerations (SAR, dB/dt, PNS)

## Your Teaching Style:

- **Educational Focus**: Explain concepts clearly with appropriate depth
- **Visual Descriptions**: Describe physics using clear analogies and examples
- **Mathematical Context**: Include relevant equations when helpful
- **Practical Connections**: Link theory to Pulseq implementation
- **Progressive Complexity**: Start simple, add detail as needed
- **Safety Awareness**: Always mention relevant safety considerations

## What You DON'T Handle:

- Pulseq syntax and programming implementation details
- Code debugging and software troubleshooting
- Language-specific conversion (MATLAB/Python)
- Documentation searches and code examples

## Response Guidelines:

- Start with fundamental concepts before advanced topics
- Use clear section headers for organization
- Include practical implications for sequence design
- Mention relevant Pulseq parameters when applicable
- Provide educational context that helps understanding
- Always consider safety implications of your explanations

Your goal is to help users understand the physics behind their Pulseq sequences, enabling better sequence design and troubleshooting."""

# Create MRI Expert agent
mri_expert_agent = Agent(
    get_llm_model(),
    deps_type=MRIExpertDependencies,
    system_prompt=MRI_EXPERT_SYSTEM_PROMPT,
)


async def consult_mri_expert(
    question: str, 
    context: str = None,
    conversation_history: list = None,
    parent_usage=None
) -> str:
    """
    Consult MRI Expert for physics explanations.
    
    Args:
        question: Physics question or topic to explain
        context: Additional context from Pulsepal conversation
        conversation_history: Recent conversation for continuity
        parent_usage: Usage tracking from parent agent
        
    Returns:
        str: Expert physics explanation
    """
    try:
        # Create dependencies for MRI Expert
        deps = MRIExpertDependencies(
            parent_usage=parent_usage
        )
        
        # Enhance question with context if provided
        enhanced_question = question
        if context:
            enhanced_question = f"Context: {context}\n\nQuestion: {question}"
        
        if conversation_history:
            history_summary = "\n".join([
                f"{entry.get('role', 'unknown')}: {entry.get('content', '')[:200]}..."
                for entry in conversation_history[-3:]  # Last 3 entries
            ])
            enhanced_question = f"Recent conversation:\n{history_summary}\n\n{enhanced_question}"
        
        # Run MRI Expert agent
        result = await mri_expert_agent.run(enhanced_question, deps=deps)
        
        logger.info("MRI Expert provided physics consultation")
        return result.data
        
    except Exception as e:
        error_msg = f"Error consulting MRI Expert: {e}"
        logger.error(error_msg)
        return f"I apologize, but I encountered an error while consulting the MRI Expert: {error_msg}. Please try asking your physics question again."


async def explain_mri_concept(concept: str, detail_level: str = "intermediate") -> str:
    """
    Get explanation of specific MRI concept with controlled detail level.
    
    Args:
        concept: MRI concept to explain (e.g., "T1 relaxation", "k-space")
        detail_level: "basic", "intermediate", or "advanced"
        
    Returns:
        str: Concept explanation at appropriate level
    """
    detail_prompts = {
        "basic": "Explain this concept in simple terms suitable for beginners:",
        "intermediate": "Provide a comprehensive explanation suitable for graduate students:",
        "advanced": "Give an advanced, detailed explanation including mathematical formulation:"
    }
    
    prompt = detail_prompts.get(detail_level, detail_prompts["intermediate"])
    question = f"{prompt} {concept}"
    
    return await consult_mri_expert(question)


async def analyze_sequence_physics(sequence_description: str, focus_area: str = None) -> str:
    """
    Analyze the physics behind a specific sequence type.
    
    Args:
        sequence_description: Description of the MRI sequence
        focus_area: Specific physics area to focus on (optional)
        
    Returns:
        str: Physics analysis of the sequence
    """
    question = f"Analyze the physics behind this MRI sequence: {sequence_description}"
    
    if focus_area:
        question += f"\n\nPlease focus specifically on: {focus_area}"
    
    return await consult_mri_expert(question)