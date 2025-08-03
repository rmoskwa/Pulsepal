#!/usr/bin/env python3
"""
Chainlit web interface for Pulsepal PydanticAI multi-agent system.

Provides a modern web UI that directly integrates with the existing PydanticAI agents,
session management, and RAG capabilities without requiring API changes.
"""

import asyncio
import logging
import uuid
from typing import Optional, List
import os

import chainlit as cl
from chainlit import Message

# Import existing Pulsepal components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from pulsepal.main_agent import pulsepal_agent, create_pulsepal_session
from pulsepal.dependencies import PulsePalDependencies, get_session_manager, SUPPORTED_LANGUAGES
from pulsepal.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global settings
settings = get_settings()


@cl.on_chat_start
async def start():
    """Initialize chat session with Pulsepal agent."""
    try:
        # Create new Pulsepal session
        session_id = str(uuid.uuid4())
        pulsepal_session_id, deps = await create_pulsepal_session(session_id)
        
        # Store session info in Chainlit user session
        cl.user_session.set("pulsepal_session_id", pulsepal_session_id)
        cl.user_session.set("pulsepal_deps", deps)
        
        # Get supported languages for welcome message
        lang_list = ", ".join([lang.upper() for lang in sorted(SUPPORTED_LANGUAGES.keys())])
        
        # Send welcome message
        welcome_msg = f"""🧠 **Welcome to Pulsepal - Your Intelligent MRI Programming Assistant!**

I'm an advanced AI with comprehensive knowledge of MRI physics and Pulseq programming, enhanced with access to specialized documentation when needed.

**🚀 What's New:**
- **Faster Responses**: I now answer general MRI and programming questions instantly
- **Smarter Search**: I only search documentation for specific Pulseq implementations
- **Enhanced Reasoning**: Better debugging support with step-by-step analysis

**💡 My Capabilities:**
🧪 **Code Generation**: Create sequences in {lang_list}
🐛 **Smart Debugging**: Analyze code logic and trace errors intelligently
🔄 **Language Conversion**: Transform code between different languages
⚛️ **MRI Physics**: Instant explanations of concepts, formulas, and principles
📚 **Selective Search**: Access Pulseq docs only when you need specific functions

**📝 Example Queries:**

*Instant Knowledge (no search needed):*
- "What is T1 relaxation?"
- "Explain the difference between spin echo and gradient echo"
- "How does k-space work?"
- "Debug: Why is my loop infinite?"

*Pulseq-Specific (selective search):*
- "How to use mr.makeGaussPulse?"
- "Show me the MOLLI sequence implementation"
- "What are the parameters for pypulseq.addBlock?"

💡 **Pro Tip**: I default to MATLAB unless you specify another language. Just ask naturally - I'll know when to search vs. when to use my knowledge!

What would you like to explore about MRI sequence programming?"""
        
        await cl.Message(content=welcome_msg).send()
        
        logger.info(f"Started Chainlit session with Pulsepal session: {pulsepal_session_id}")
        
    except Exception as e:
        logger.error(f"Failed to initialize chat session: {e}")
        await cl.Message(
            content=f"❌ **Error**: Failed to initialize Pulsepal session: {e}\n\nPlease refresh the page and try again.",
            author="System"
        ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages and respond using Pulsepal agent."""
    try:
        # Get session info
        pulsepal_session_id = cl.user_session.get("pulsepal_session_id")
        deps = cl.user_session.get("pulsepal_deps")
        
        if not pulsepal_session_id or not deps:
            await cl.Message(
                content="❌ **Session Error**: Please refresh the page to restart your session.",
                author="System"
            ).send()
            return
        
        
        # Show typing indicator with intelligent status
        async with cl.Step(name="🧠 Analyzing your query...") as step:
            try:
                # Add user message to conversation context
                deps.conversation_context.add_conversation("user", message.content)
                
                # Detect language preference from query
                deps.conversation_context.detect_language_preference(message.content)
                
                # Simple, direct approach - let the agent's intelligence handle everything
                result = await pulsepal_agent.run(message.content, deps=deps)
                
                # Add response to conversation history
                deps.conversation_context.add_conversation("assistant", result.data)
                
                step.output = "✅ Response ready"
                
            except Exception as e:
                logger.error(f"Error running Pulsepal agent: {e}")
                step.output = f"❌ Error: {e}"
                result_output = f"I apologize, but I encountered an error: {e}\n\nPlease try rephrasing your question or check that all services are running properly."
            else:
                result_output = result.data
        
        # Send the response with streaming effect
        msg = cl.Message(content="", author="Pulsepal")
        
        # Option 1: Stream by words (more natural)
        words = result_output.split(' ')
        for i, word in enumerate(words):
            # Add space before word (except first word)
            if i > 0:
                await msg.stream_token(' ')
            await msg.stream_token(word)
            # Variable delay based on word length for more natural feel
            delay = min(0.05, 0.01 + len(word) * 0.005)  # 10-50ms per word
            await asyncio.sleep(delay)
        
        # Option 2: Stream by characters (uncomment to use)
        # for char in result_output:
        #     await msg.stream_token(char)
        #     await asyncio.sleep(0.005)  # 5ms delay per character
        
        # Finalize the message
        await msg.update()
        
        logger.info(f"Processed message in session {pulsepal_session_id}")
        
    except Exception as e:
        logger.error(f"Critical error in message handler: {e}")
        await cl.Message(
            content=f"❌ **Critical Error**: {e}\n\nPlease refresh the page to restart your session.",
            author="System"
        ).send()


# Note: File upload functionality disabled due to Chainlit version compatibility
# The @cl.on_file_upload decorator is not available in Chainlit 2.6.4
# Users can paste code directly in messages for analysis


@cl.on_chat_end
async def end():
    """Clean up when chat session ends."""
    try:
        pulsepal_session_id = cl.user_session.get("pulsepal_session_id")
        if pulsepal_session_id:
            # Optional: Clean up expired sessions
            session_manager = get_session_manager()
            await session_manager.cleanup_expired_sessions()
            logger.info(f"Cleaned up Chainlit session for Pulsepal session: {pulsepal_session_id}")
    except Exception as e:
        logger.error(f"Error during session cleanup: {e}")


# Custom settings for Chainlit
@cl.step(type="tool")
async def rag_search_step(query: str, tool_name: str) -> str:
    """Display RAG search operations as structured steps."""
    return f"🔍 Searching {tool_name} for: {query}"


# Note: cl.on_error is not available in Chainlit 2.6.3
# Error handling is done within individual handlers


if __name__ == "__main__":
    # This won't be called when running with `chainlit run`
    # But useful for testing imports
    print("Chainlit app loaded successfully. Run with: chainlit run chainlit_app.py")