#!/usr/bin/env python3
"""
Chainlit web interface for Pulsepal PydanticAI multi-agent system.

Provides a modern web UI that directly integrates with the existing PydanticAI agents,
session management, and RAG capabilities without requiring API changes.
"""

import asyncio
import logging
import uuid
from typing import Optional
import os

import chainlit as cl

# Import existing Pulsepal components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from pulsepal.main_agent import pulsepal_agent, create_pulsepal_session
from pulsepal.dependencies import get_session_manager, SUPPORTED_LANGUAGES
from pulsepal.settings import get_settings
from pulsepal.conversation_logger import get_conversation_logger

# Configure logging early so it's available for auth
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug: Log startup information
logger.info("=" * 60)
logger.info("PULSEPAL CHAINLIT STARTUP")
logger.info(f"Python path: {sys.path}")
logger.info(f"Current directory: {os.getcwd()}")
logger.info(f"Environment ALPHA_API_KEYS exists: {'ALPHA_API_KEYS' in os.environ}")
if 'ALPHA_API_KEYS' in os.environ:
    # Don't log the actual keys for security
    logger.info(f"ALPHA_API_KEYS length: {len(os.environ['ALPHA_API_KEYS'])}")
logger.info("=" * 60)

# Optional authentication for deployment
AUTH_ENABLED = False
try:
    logger.info("Attempting to import auth module...")
    from pulsepal.auth import check_rate_limit, validate_api_key, API_KEYS
    logger.info(f"Auth module imported successfully. API_KEYS type: {type(API_KEYS)}, length: {len(API_KEYS)}")
    logger.info(f"API key names: {list(API_KEYS.keys())}")
    
    # Only enable auth if we have real API keys (not just test-key)
    if len(API_KEYS) > 0 and not (len(API_KEYS) == 1 and "test-key" in API_KEYS):
        AUTH_ENABLED = True
        logger.info(f"🔐 AUTHENTICATION ENABLED with {len(API_KEYS)} API keys")
        
        # CRITICAL: Register auth callback HERE, before any other Chainlit decorators
        @cl.password_auth_callback
        def auth_callback(username: str, password: str) -> Optional[cl.User]:
            """Authenticate user with API key as password."""
            logger.info(f"Auth callback called with username: {username}")
            user_info = validate_api_key(password)
            
            if user_info:
                logger.info(f"✅ Successful login for: {username or user_info['email']}")
                return cl.User(
                    identifier=username or user_info["email"],
                    metadata={
                        "api_key": password,
                        "name": user_info["name"],
                        "limit": user_info.get("limit", 100)
                    }
                )
            else:
                logger.warning(f"❌ Failed login attempt with username: {username}")
            return None
            
        logger.info("Auth callback registered successfully")
    else:
        logger.info("🔓 Authentication disabled - no valid API keys configured")
        logger.info(f"Reason: len(API_KEYS)={len(API_KEYS)}, has only test-key={len(API_KEYS) == 1 and 'test-key' in API_KEYS}")
        
except ImportError as e:
    # Auth module not available
    logger.error(f"❌ Auth module import failed: {e}")
    import traceback
    logger.error(traceback.format_exc())
    AUTH_ENABLED = False
except Exception as e:
    logger.error(f"❌ Unexpected error during auth setup: {e}")
    import traceback
    logger.error(traceback.format_exc())
    AUTH_ENABLED = False
    
# Define fallback rate limiting if auth is disabled
if not AUTH_ENABLED:
    def check_rate_limit(api_key: str, limit: int = 100) -> bool:
        """Dummy rate limit function when auth is disabled."""
        return True

# Final auth status
logger.info("=" * 60)
logger.info(f"🔒 FINAL AUTH STATUS: AUTH_ENABLED = {AUTH_ENABLED}")
logger.info("=" * 60)

# Global settings
settings = get_settings()

# Initialize conversation logger for debugging
conversation_logger = get_conversation_logger()


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
        
        # Log session start for debugging
        conversation_logger.log_conversation(
            pulsepal_session_id,
            "system",
            "Session started",
            {"event": "session_start"}
        )
        
        # Get supported languages for welcome message
        lang_list = ", ".join([lang.upper() for lang in sorted(SUPPORTED_LANGUAGES.keys())])
        
        # Get user info if authenticated
        auth_info = ""
        if AUTH_ENABLED:
            user = cl.user_session.get("user")
            logger.info(f"User session in on_chat_start: {user}")
            if user:
                logger.info(f"User metadata: {user.metadata if hasattr(user, 'metadata') else 'No metadata'}")
                if hasattr(user, 'metadata') and user.metadata:
                    user_name = user.metadata.get("name", "User")
                    user_limit = user.metadata.get("limit", 100)
                    auth_info = f"\n\n👤 **Welcome back, {user_name}!**\n📊 Rate limit: {user_limit} requests/hour"
                    logger.info(f"Generated auth_info: {auth_info}")
                else:
                    logger.warning("User found but no metadata available")
            else:
                logger.warning("No user found in session during on_chat_start")
        
        # Send welcome message
        welcome_msg = f"""🧠 **Welcome to Pulsepal - Your Intelligent MRI Programming Assistant!**{auth_info}

I'm an advanced AI with comprehensive knowledge of MRI physics and Pulseq programming, enhanced with access to specialized documentation when needed.

**🚀 What's New:**
- **Code Upload Support**: Drag and drop your .m or .py files for debugging help
- **Faster Responses**: I now answer general MRI and programming questions instantly
- **Smarter Search**: I only search documentation for specific Pulseq implementations
- **Enhanced Reasoning**: Better debugging support with step-by-step analysis

**💡 My Capabilities:**
🧪 **Code Generation**: Create sequences in {lang_list}
🐛 **Smart Debugging**: Analyze code logic and trace errors intelligently (upload your files!)
🔄 **Language Conversion**: Transform code between different languages
⚛️ **MRI Physics**: Instant explanations of concepts, formulas, and principles
📚 **Selective Search**: Access Pulseq docs only when you need specific functions
📎 **File Upload**: Attach .m or .py files to share your code for debugging

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

📋 **Commands**: Type `/info` to see your account information and rate limits.

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
        # Extract code from uploaded files if present
        code_context = ""
        if message.elements:
            for element in message.elements:
                # Check if it's a file element with .m or .py extension
                if hasattr(element, 'name') and element.name.endswith(('.m', '.py')):
                    try:
                        # Get file content - handle both bytes and string content
                        if hasattr(element, 'content'):
                            file_content = element.content.decode('utf-8') if isinstance(element.content, bytes) else element.content
                        elif hasattr(element, 'path'):
                            # For local file uploads, read from path
                            with open(element.path, 'r', encoding='utf-8') as f:
                                file_content = f.read()
                        else:
                            logger.warning(f"Could not extract content from file: {element.name}")
                            continue
                        
                        # Check file size (limit to 1MB)
                        if len(file_content) > 1024 * 1024:
                            await cl.Message(
                                content=f"⚠️ File '{element.name}' is too large (>1MB). Please reduce the file size.",
                                author="System"
                            ).send()
                            continue
                        
                        # Add code to context
                        code_context += f"\n\n--- Code from {element.name} ---\n{file_content}\n--- End of {element.name} ---\n"
                        
                        # Send confirmation to user
                        await cl.Message(content=f"✅ Loaded file: {element.name}").send()
                        logger.info(f"Successfully loaded file: {element.name} ({len(file_content)} bytes)")
                        
                    except Exception as e:
                        logger.error(f"Error loading file {element.name}: {e}")
                        await cl.Message(
                            content=f"❌ Error loading file '{element.name}': {e}",
                            author="System"
                        ).send()
        
        # Combine user query with code context
        enhanced_query = message.content
        if code_context:
            enhanced_query = f"{message.content}\n\nHere is my code:{code_context}"
            logger.info(f"Enhanced query with {len(code_context)} bytes of code context")
        
        # Special command to check user info
        if message.content.strip().lower() == "/info":
            user = cl.user_session.get("user")
            if AUTH_ENABLED and user and hasattr(user, 'metadata') and user.metadata:
                user_name = user.metadata.get("name", "Unknown")
                user_email = user.identifier
                user_limit = user.metadata.get("limit", 100)
                api_key = user.metadata.get("api_key", "Unknown")
                
                info_msg = f"""📋 **Your Account Information**
                
👤 **Name**: {user_name}
📧 **Email**: {user_email}
🔑 **API Key**: {api_key[:8]}...
📊 **Rate Limit**: {user_limit} requests/hour
🔒 **Authentication**: Enabled"""
            else:
                info_msg = "🔓 **Authentication**: Disabled (local mode)"
            
            await cl.Message(content=info_msg).send()
            return
        
        # Check rate limiting for authenticated users (if auth is enabled)
        if AUTH_ENABLED:
            user = cl.user_session.get("user")
            if user and hasattr(user, 'metadata') and user.metadata:
                api_key = user.metadata.get("api_key")
                limit = user.metadata.get("limit", 100)
                user_name = user.metadata.get("name", "User")
                
                if api_key and not check_rate_limit(api_key, limit):
                    logger.warning(f"Rate limit exceeded for user: {user_name} (API key: {api_key[:8]}...)")
                    await cl.Message(
                        content=f"⚠️ **Rate limit exceeded**\n\nHi {user_name}, you've reached your limit of {limit} requests per hour.\n\nPlease wait a few minutes before sending more requests.",
                        author="System"
                    ).send()
                    return
                elif api_key:
                    logger.debug(f"Rate limit check passed for user: {user_name}")
        
        # Get session info
        pulsepal_session_id = cl.user_session.get("pulsepal_session_id")
        deps = cl.user_session.get("pulsepal_deps")
        
        if not pulsepal_session_id or not deps:
            await cl.Message(
                content="❌ **Session Error**: Please refresh the page to restart your session.",
                author="System"
            ).send()
            return
        
        # Debug logging for session continuity
        logger.debug(f"Session {pulsepal_session_id} - History length: {len(deps.conversation_context.conversation_history)}")
        if deps.conversation_context.conversation_history:
            last_entry = deps.conversation_context.conversation_history[-1]
            logger.debug(f"Session {pulsepal_session_id} - Last message: {last_entry['role']}: {last_entry['content'][:50]}...")
        
        # Show typing indicator with intelligent status
        async with cl.Step(name="🧠 Analyzing your query...") as step:
            try:
                # Add user message to conversation context (use enhanced_query if code was uploaded)
                deps.conversation_context.add_conversation("user", enhanced_query)
                
                # Log user message for debugging
                conversation_logger.log_conversation(
                    pulsepal_session_id,
                    "user",
                    enhanced_query
                )
                
                # Get conversation history for context
                history_context = deps.conversation_context.get_formatted_history()
                
                # Create query with context
                if history_context:
                    query_with_context = f"{history_context}\n\nCurrent query: {enhanced_query}"
                else:
                    query_with_context = enhanced_query
                
                # Detect language preference from query
                deps.conversation_context.detect_language_preference(enhanced_query)
                
                # Run agent with query including context
                result = await pulsepal_agent.run(query_with_context, deps=deps)
                
                # Add response to conversation history
                deps.conversation_context.add_conversation("assistant", result.data)
                
                # Log assistant response for debugging
                conversation_logger.log_conversation(
                    pulsepal_session_id,
                    "assistant",
                    result.data
                )
                
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
        
    except asyncio.TimeoutError:
        logger.warning("Message processing timed out")
        await cl.Message(
            content=(
                "⏱️ **Request Timed Out**\n\n"
                "The search took too long to complete. Please try:\n"
                "- Using a more specific query\n"
                "- Breaking your question into smaller parts\n"
                "- Asking about general concepts instead of searching for specific implementations"
            ),
            author="System"
        ).send()
    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        await cl.Message(
            content=(
                "🔌 **Connection Error**\n\n"
                "I'm having trouble connecting to the knowledge base. "
                "Please check your internet connection and try again.\n\n"
                "I can still help with general MRI physics and Pulseq concepts using my built-in knowledge."
            ),
            author="System"
        ).send()
    except Exception as e:
        error_type = type(e).__name__
        logger.error(f"Critical error in message handler ({error_type}): {e}")
        
        # Provide user-friendly error messages
        if "supabase" in str(e).lower() or "database" in str(e).lower():
            error_msg = (
                "🗄️ **Knowledge Base Unavailable**\n\n"
                "I'm experiencing issues accessing the Pulseq knowledge base. "
                "I can still help with general MRI physics and sequence concepts using my built-in knowledge.\n\n"
                "Try asking questions about:\n"
                "- MRI physics principles\n"
                "- Sequence timing and structure\n"
                "- General Pulseq concepts"
            )
        elif "rate" in str(e).lower() and "limit" in str(e).lower():
            error_msg = (
                "⚠️ **Rate Limit Reached**\n\n"
                "You've reached the maximum number of requests. "
                "Please wait a moment before trying again."
            )
        else:
            error_msg = (
                "❌ **Unexpected Error**\n\n"
                "I encountered an error while processing your request. Please try:\n"
                "- Rephrasing your question\n"
                "- Breaking it into smaller parts\n"
                "- Asking about general concepts instead of specific implementations\n\n"
                f"Error details: {error_type}"
            )
        
        await cl.Message(
            content=error_msg,
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
            # Log session end for debugging
            conversation_logger.log_conversation(
                pulsepal_session_id,
                "system",
                "Session ended",
                {"event": "session_end"}
            )
            
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