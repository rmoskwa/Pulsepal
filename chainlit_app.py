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
        welcome_msg = f"""🔬 **Welcome to Pulsepal!**

I'm your AI assistant for Pulseq MRI sequence programming. I can help you with:

🧪 **Code Generation**: Create sequences in multiple programming languages
🐛 **Debugging**: Fix sequence errors and optimization issues  
🔄 **Language Conversion**: Convert between different programming languages
📚 **Documentation**: Search comprehensive Pulseq documentation
⚛️ **Physics**: Get expert explanations of MRI physics concepts
📁 **File Analysis**: Upload and analyze sequence files

**Supported Languages:**
{lang_list}

**Quick Examples:**
- "How do I create a spin echo sequence in MATLAB?"
- "Convert this MATLAB sequence to Python"
- "Show me a C++ implementation of gradient echo"
- "Explain k-space trajectory for EPI sequences"
- "Help me debug this Julia sequence code"

💡 **Tip**: I default to MATLAB code examples unless you specify another language!

📎 **Code Analysis**: Paste code snippets directly in your messages for analysis and debugging.

What would you like to know about Pulseq programming?"""
        
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
        
        
        # Show typing indicator
        async with cl.Step(name="🔬 Pulsepal is thinking...") as step:
            try:
                # Add user message to conversation context
                deps.conversation_context.add_conversation("user", message.content)
                
                # Detect language preference if needed
                if not deps.conversation_context.preferred_language or deps.conversation_context.preferred_language == "matlab":
                    deps.conversation_context.detect_language_preference(message.content)
                
                # Prepare context-aware query
                enhanced_query = message.content
                if deps.conversation_context.conversation_history:
                    # Get recent conversation history for context (last 3 exchanges)
                    recent_history = deps.conversation_context.get_recent_conversations(6)  # 3 exchanges = 6 messages
                    if len(recent_history) > 2:  # Only add context if there's meaningful history
                        context_summary = "Recent conversation context:\n"
                        for entry in recent_history[-6:-1]:  # Exclude the current message
                            role = entry.get('role', 'unknown')
                            content = entry.get('content', '')[:100]  # Limit for context
                            context_summary += f"{role}: {content}...\n"
                        
                        enhanced_query = f"{context_summary}\nCurrent question: {message.content}"
                
                # Include preferred language context
                preferred_lang = deps.conversation_context.preferred_language or "matlab"
                enhanced_query += f"\n\nUser's preferred programming language: {preferred_lang}"
                
                # Run Pulsepal agent
                result = await pulsepal_agent.run(enhanced_query, deps=deps)
                
                # Add response to conversation history
                deps.conversation_context.add_conversation("assistant", result.output)
                
                step.output = "✅ Response generated successfully"
                
            except Exception as e:
                logger.error(f"Error running Pulsepal agent: {e}")
                step.output = f"❌ Error: {e}"
                result_output = f"I apologize, but I encountered an error: {e}\n\nPlease try rephrasing your question or check that all services are running properly."
            else:
                result_output = result.output
        
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