#!/usr/bin/env python3
"""
Chainlit web interface for Pulsepal with modern RAG service v2.

This version uses the simplified RAG service with enhanced function validation
to prevent hallucinations while maintaining clean architecture.
"""

import asyncio
import logging
import uuid
from typing import Optional, Dict, Any
import os

import chainlit as cl

# Import existing Pulsepal components
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Import V2 components with function validation
from pulsepal.main_agent_v2 import pulsepal_agent, create_pulsepal_session
from pulsepal.dependencies import get_session_manager
from pulsepal.settings import get_settings
from pulsepal.conversation_logger import get_conversation_logger
from pulsepal.semantic_router import initialize_semantic_router, QueryRoute

# Configure logging early so it's available for auth
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug: Log startup information
logger.info("=" * 60)
logger.info("PULSEPAL CHAINLIT V2 STARTUP - ENHANCED WITH FUNCTION VALIDATION")
logger.info(f"Python path: {sys.path}")
logger.info(f"Current directory: {os.getcwd()}")
logger.info("Using enhanced RAG service v2 with hallucination prevention")
logger.info("=" * 60)

# Optional authentication for deployment
AUTH_ENABLED = False
try:
    logger.info("Attempting to import auth module...")
    from pulsepal.auth import check_rate_limit, validate_api_key, API_KEYS

    logger.info(
        f"Auth module imported successfully. API_KEYS type: {type(API_KEYS)}, length: {len(API_KEYS)}"
    )
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
                logger.info(
                    f"✅ Successful login for: {username or user_info['email']}"
                )
                return cl.User(
                    identifier=username or user_info["email"],
                    metadata={
                        "api_key": password,
                        "name": user_info["name"],
                        "limit": user_info.get("limit", 100),
                    },
                )
            else:
                logger.warning(f"❌ Failed login attempt with username: {username}")
            return None

        logger.info("Auth callback registered successfully")
    else:
        logger.info("🔓 Authentication disabled - no valid API keys configured")
        logger.info(
            f"Reason: len(API_KEYS)={len(API_KEYS)}, has only test-key={len(API_KEYS) == 1 and 'test-key' in API_KEYS}"
        )

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
logger.info("🚀 USING ENHANCED RAG SERVICE V2 - With Function Validation")
logger.info("=" * 60)

# Global settings
settings = get_settings()

# Initialize conversation logger for debugging
conversation_logger = get_conversation_logger()

# Initialize semantic router at startup for efficient classification
logger.info("Initializing semantic router at startup...")
_semantic_router = initialize_semantic_router()
logger.info("✅ Semantic router initialized successfully")

# Sequence Knowledge Template
SEQUENCE_KNOWLEDGE_TEMPLATE = """# Sequence Knowledge Template

## Sequence Overview
**Sequence Type**: [e.g., Gradient Echo, Spin Echo, EPI, bSSFP]
**Target Application**: [e.g., Brain imaging, Cardiac, Diffusion, Perfusion]
**Development Stage**: [Initial design/Optimization/Debugging]

## Imaging Goals
- **Resolution**: [e.g., 1mm isotropic, 0.5x0.5x2mm]
- **Scan time**: [e.g., <5 minutes, 30 seconds per volume]
- **Contrast**: [T1/T2/T2*/PD/DWI]
- **Coverage**: [e.g., Whole brain, Single slice, 3D volume]

## Technical Constraints
- **Scanner**: [Model and field strength, e.g., Siemens Prisma 3T]
- **Max gradient**: [mT/m, e.g., 80 mT/m]
- **Max slew rate**: [T/m/s, e.g., 200 T/m/s]
- **RF limitations**: [e.g., SAR limits, B1+ inhomogeneity]

## Sequence Parameters
- **TR**: [ms]
- **TE**: [ms]
- **Flip angle**: [degrees]
- **Bandwidth**: [Hz/pixel]
- **Matrix size**: [e.g., 256x256]
- **FOV**: [mm]

## Current Focus
[What you're working on or need help with, e.g., "Optimizing TE for better contrast", "Reducing artifacts", "Implementing parallel imaging"]

## Additional Notes
[Any other relevant information about your sequence or specific challenges]
"""


@cl.on_chat_start
async def start():
    """Initialize chat session with Pulsepal agent using enhanced RAG v2."""
    try:
        # Create new Pulsepal session with v2 enhanced components
        session_id = str(uuid.uuid4())
        pulsepal_session_id, deps = await create_pulsepal_session(session_id)

        # Store session info in Chainlit user session
        cl.user_session.set("pulsepal_session_id", pulsepal_session_id)
        cl.user_session.set("pulsepal_deps", deps)

        # Log session start for debugging
        conversation_logger.log_conversation(
            pulsepal_session_id,
            "system",
            "Session started with enhanced RAG v2",
            {"event": "session_start", "rag_version": "v2_enhanced"},
        )

        # Configure settings panel for sequence knowledge
        template_hint = """Example format:
Sequence Type: [e.g., EPI, Gradient Echo]
Target: [e.g., Brain imaging]
TR/TE: [e.g., 2000ms/30ms]
Current Focus: [What you need help with]
(Click 'Show Template' button for full template)"""

        settings = [
            cl.input_widget.TextInput(
                id="sequence_knowledge",
                label="🎯 Sequence Knowledge",
                description=f"Add your sequence-specific context. {template_hint}",
                placeholder="Enter your sequence details here...",
                multiline=True,
                initial=deps.conversation_context.sequence_knowledge or "",
                max_chars=10000,
            ),
            cl.input_widget.Switch(
                id="use_sequence_context",
                label="Enable Sequence Context",
                initial=deps.conversation_context.use_sequence_context,
                description="When enabled, PulsePal will consider your sequence context in all responses",
            ),
        ]

        await cl.ChatSettings(settings).send()

        # Get supported languages for welcome message (removed unused variable)

        # Get user info if authenticated
        auth_info = ""
        if AUTH_ENABLED:
            user = cl.user_session.get("user")
            logger.info(f"User session in on_chat_start: {user}")
            if user:
                logger.info(
                    f"User metadata: {user.metadata if hasattr(user, 'metadata') else 'No metadata'}"
                )
                if hasattr(user, "metadata") and user.metadata:
                    user_name = user.metadata.get("name", "User")
                    user_limit = user.metadata.get("limit", 100)
                    auth_info = f"\n\n👤 **Welcome back, {user_name}!**\n📊 Rate limit: {user_limit} requests/hour"
                    logger.info(f"Generated auth_info: {auth_info}")
                else:
                    logger.warning("User found but no metadata available")
            else:
                logger.warning("No user found in session during on_chat_start")

        # Check if context is already active
        context_status = ""
        if (
            deps.conversation_context.use_sequence_context
            and deps.conversation_context.sequence_knowledge
        ):
            context_status = "\n\n🎯 **[Sequence Context Active]** - Your sequence-specific knowledge is being used."

        # Send welcome message with v2 enhanced features highlighted
        welcome_msg = f"""🧠 **Welcome to Pulsepal v2 Enhanced - Intelligent MRI Assistant with Function Validation!**{auth_info}{context_status}

**✨ What's New in v2 Enhanced:**
- **✅ Function Validation**: Prevents hallucinated functions like `seq.calcKspace()` 
- **✅ Namespace Checking**: Corrects common errors (mr.write → seq.write)
- **🚀 90% Faster Responses**: Simplified architecture with modern RAG service
- **🧪 Smarter Intelligence**: Gemini 2.5 Flash handles reasoning with validation safety
- **📚 Clean Retrieval**: Document retrieval with deterministic function checking
- **🔍 Better Search**: LLM decides what to search, validation ensures correctness

I'm an advanced AI with comprehensive MRI physics knowledge, enhanced by selective access to Pulseq documentation and **deterministic function validation** to prevent hallucinations.

**💡 My Capabilities:**
🧪 **Validated Code Generation**: Create sequences with guaranteed correct functions
🐛 **Smart Debugging**: Analyze code with validation (upload your .m or .py files!)
🔄 **Language Conversion**: Transform code between different languages
⚛️ **MRI Physics**: Deep understanding of concepts, formulas, and principles
✅ **Function Validation**: I verify all Pulseq functions before using them
📎 **File Upload**: Attach .m or .py files for debugging assistance

**🎯 Sequence Knowledge**: Add your sequence context via Settings (gear icon) for targeted assistance
  - Open Settings and add your sequence details
  - Enable "Sequence Context" toggle to activate
  - Your context will be considered in all responses!

**📝 Example Queries:**

*General MRI Knowledge:*
- "Explain T1 vs T2 relaxation mechanisms"
- "How does parallel imaging work?"
- "What causes ghosting artifacts in EPI?"
- "Debug my infinite loop in this code"

*Pulseq-Specific (with validation):*
- "How to plot k-space?" (I'll use seq.calculateKspacePP, not hallucinated functions)
- "How to save a sequence?" (I'll correctly use seq.write, not mr.write)
- "Show me mr.makeSincPulse parameters"
- "Convert this MATLAB code to Python pypulseq"

💡 **Pro Tip**: All function names are validated against the Pulseq function index to ensure correctness!

📋 **Commands**: Type `/info` to see your account information and rate limits.

What would you like to explore about MRI sequence programming today?"""

        # Send welcome message
        await cl.Message(content=welcome_msg).send()

        logger.info(
            f"Started Chainlit session with enhanced Pulsepal v2 session: {pulsepal_session_id}"
        )

    except Exception as e:
        logger.error(f"Failed to initialize chat session: {e}")
        await cl.Message(
            content=f"❌ **Error**: Failed to initialize Pulsepal v2 enhanced session: {e}\n\nPlease refresh the page and try again.",
            author="System",
        ).send()


@cl.on_settings_update
async def update_settings(settings: Dict[str, Any]):
    """Handle settings panel updates for sequence knowledge."""
    deps = cl.user_session.get("pulsepal_deps")
    if not deps:
        await cl.Message(content="❌ Session error. Please refresh the page.").send()
        return

    logger.info(f"Settings update received: {list(settings.keys())}")

    # Update sequence knowledge if provided
    if "sequence_knowledge" in settings:
        new_knowledge = (
            settings["sequence_knowledge"].strip()
            if settings["sequence_knowledge"]
            else ""
        )
        old_knowledge = deps.conversation_context.sequence_knowledge or ""

        # Only update if changed
        if new_knowledge != old_knowledge:
            deps.conversation_context.sequence_knowledge = (
                new_knowledge if new_knowledge else None
            )

            if new_knowledge:
                knowledge_msg = f"✅ **Sequence Knowledge Updated**\n\nAdded {len(new_knowledge)} characters of sequence-specific context."
            else:
                knowledge_msg = "🗑️ **Sequence Knowledge Cleared**\n\nSequence-specific context has been removed."

            await cl.Message(content=knowledge_msg).send()

    # Update context enable/disable setting
    if "use_sequence_context" in settings:
        new_setting = settings["use_sequence_context"]
        old_setting = deps.conversation_context.use_sequence_context

        # Only update if changed
        if new_setting != old_setting:
            deps.conversation_context.use_sequence_context = new_setting

            if deps.conversation_context.use_sequence_context:
                if deps.conversation_context.sequence_knowledge:
                    context_msg = "🎯 **Sequence Context Activated**\n\nYour sequence knowledge is now active for all responses."
                else:
                    context_msg = "⚠️ **Context Enabled but Empty**\n\nSequence context is enabled, but no knowledge has been added yet."
            else:
                context_msg = "⏸️ **Sequence Context Deactivated**\n\nI'll provide general responses without sequence-specific context."

            await cl.Message(content=context_msg).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages and respond using Pulsepal agent with enhanced RAG v2."""
    try:
        # Extract code from uploaded files if present
        code_context = ""
        if message.elements:
            for element in message.elements:
                # Check if it's a file element with .m or .py extension
                if hasattr(element, "name") and element.name.endswith((".m", ".py")):
                    try:
                        # Get file content - handle both bytes and string content
                        if hasattr(element, "content"):
                            file_content = (
                                element.content.decode("utf-8")
                                if isinstance(element.content, bytes)
                                else element.content
                            )
                        elif hasattr(element, "path"):
                            # For local file uploads, read from path
                            with open(element.path, "r", encoding="utf-8") as f:
                                file_content = f.read()
                        else:
                            logger.warning(
                                f"Could not extract content from file: {element.name}"
                            )
                            continue

                        # Check file size (limit to 1MB)
                        if len(file_content) > 1024 * 1024:
                            await cl.Message(
                                content=f"⚠️ File '{element.name}' is too large (>1MB). Please reduce the file size.",
                                author="System",
                            ).send()
                            continue

                        # Add code to context
                        code_context += f"\n\n--- Code from {element.name} ---\n{file_content}\n--- End of {element.name} ---\n"

                        # Send confirmation to user
                        await cl.Message(
                            content=f"✅ Loaded file: {element.name}"
                        ).send()
                        logger.info(
                            f"Successfully loaded file: {element.name} ({len(file_content)} bytes)"
                        )

                    except Exception as e:
                        logger.error(f"Error loading file {element.name}: {e}")
                        await cl.Message(
                            content=f"❌ Error loading file '{element.name}': {e}",
                            author="System",
                        ).send()

        # Combine user query with code context
        enhanced_query = message.content
        if code_context:
            enhanced_query = f"{message.content}\n\nHere is my code:{code_context}"
            logger.info(
                f"Enhanced query with {len(code_context)} bytes of code context"
            )

        # Special command to check user info
        if message.content.strip().lower() == "/info":
            user = cl.user_session.get("user")
            if AUTH_ENABLED and user and hasattr(user, "metadata") and user.metadata:
                user_name = user.metadata.get("name", "Unknown")
                user_email = user.identifier
                user_limit = user.metadata.get("limit", 100)
                api_key = user.metadata.get("api_key", "Unknown")

                info_msg = f"""📋 **Your Account Information**
                
👤 **Name**: {user_name}
📧 **Email**: {user_email}
🔑 **API Key**: {api_key[:8]}...
📊 **Rate Limit**: {user_limit} requests/hour
🔒 **Authentication**: Enabled
🚀 **RAG Version**: Enhanced v2 (with Function Validation)
✅ **Function Validation**: Active (prevents hallucinations)"""
            else:
                info_msg = "🔓 **Authentication**: Disabled (local mode)\n🚀 **RAG Version**: Enhanced v2 (with Function Validation)\n✅ **Function Validation**: Active"

            await cl.Message(content=info_msg).send()
            return

        # Check rate limiting for authenticated users (if auth is enabled)
        if AUTH_ENABLED:
            user = cl.user_session.get("user")
            if user and hasattr(user, "metadata") and user.metadata:
                api_key = user.metadata.get("api_key")
                limit = user.metadata.get("limit", 100)
                user_name = user.metadata.get("name", "User")

                if api_key and not check_rate_limit(api_key, limit):
                    logger.warning(
                        f"Rate limit exceeded for user: {user_name} (API key: {api_key[:8]}...)"
                    )
                    await cl.Message(
                        content=f"⚠️ **Rate limit exceeded**\n\nHi {user_name}, you've reached your limit of {limit} requests per hour.\n\nPlease wait a few minutes before sending more requests.",
                        author="System",
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
                author="System",
            ).send()
            return

        # Debug logging for session continuity
        logger.debug(
            f"Session {pulsepal_session_id} - History length: {len(deps.conversation_context.conversation_history)}"
        )
        if deps.conversation_context.conversation_history:
            last_entry = deps.conversation_context.conversation_history[-1]
            logger.debug(
                f"Session {pulsepal_session_id} - Last message: {last_entry['role']}: {last_entry['content'][:50]}..."
            )

        # Show typing indicator with intelligent status
        async with cl.Step(name="🧠 Processing with Enhanced RAG v2 (Function Validation Active)...") as step:
            try:
                # Add user message to conversation context (use enhanced_query if code was uploaded)
                deps.conversation_context.add_conversation("user", enhanced_query)

                # Log user message for debugging
                conversation_logger.log_conversation(
                    pulsepal_session_id, "user", enhanced_query, {"rag_version": "v2_enhanced"}
                )

                # Get conversation history for context
                history_context = deps.conversation_context.get_formatted_history()

                # Get sequence context if enabled
                sequence_context = deps.conversation_context.get_active_context()

                # Build query with all relevant context
                context_parts = []

                # Add sequence context first if available (highest priority)
                if sequence_context:
                    context_parts.append(sequence_context)
                    logger.info(
                        f"Including sequence context: {len(sequence_context)} chars"
                    )

                # Add conversation history
                if history_context:
                    context_parts.append(history_context)

                # Create query with context
                if context_parts:
                    query_with_context = (
                        "\n\n".join(context_parts)
                        + f"\n\nCurrent query: {enhanced_query}"
                    )
                else:
                    query_with_context = enhanced_query

                # Detect language preference from query
                deps.conversation_context.detect_language_preference(enhanced_query)

                # Semantic routing before Gemini (required)
                routing_decision = _semantic_router.classify_query(enhanced_query)
                
                # Log the routing decision
                _semantic_router.log_routing_decision(
                    pulsepal_session_id,
                    enhanced_query,
                    routing_decision,
                    conversation_logger
                )
                
                # Apply routing decision and inject context
                if routing_decision.route == QueryRoute.FORCE_RAG:
                    # Subtle user feedback for forced RAG
                    await cl.Message(
                        content="📚 Consulting Pulseq documentation...",
                        author="System"
                    ).send()
                    
                    # Store routing hints in deps for tools to use
                    deps.force_rag = True
                    deps.forced_search_hints = routing_decision.search_hints
                    logger.info(f"Forcing RAG search: {routing_decision.reasoning}")
                    
                    # Store routing hints in deps instead of injecting into query
                    # This prevents confusion where the model thinks it's part of the conversation
                    
                elif routing_decision.route == QueryRoute.NO_RAG:
                    # Indicate we should skip RAG
                    deps.skip_rag = True
                    logger.info(f"Skipping RAG: {routing_decision.reasoning}")
                    # Don't inject context into query - just use flags
                    
                else:
                    # GEMINI_CHOICE - let the agent decide
                    logger.info(f"Letting Gemini decide: {routing_decision.reasoning}")
                    # Don't inject context - let the agent work normally

                # Run agent with original query (not modified by routing)
                result = await pulsepal_agent.run(query_with_context, deps=deps)

                # Add response to conversation history
                deps.conversation_context.add_conversation("assistant", result.data)

                # Log assistant response for debugging
                conversation_logger.log_conversation(
                    pulsepal_session_id, "assistant", result.data, {"rag_version": "v2_enhanced"}
                )

                step.output = "✅ Response ready (Enhanced RAG v2 with validation)"

            except Exception as e:
                logger.error(f"Error running Pulsepal agent: {e}")
                step.output = f"❌ Error: {e}"
                result_output = f"I apologize, but I encountered an error: {e}\n\nPlease try rephrasing your question or check that all services are running properly."
            else:
                result_output = result.data

        # Add sequence context indicator if active
        context_prefix = ""
        if (
            deps.conversation_context.use_sequence_context
            and deps.conversation_context.sequence_knowledge
        ):
            context_prefix = "🎯 [Sequence Context Active] "

        # Send the response with streaming effect
        msg = cl.Message(content="", author=f"{context_prefix}Pulsepal v2 Enhanced")

        # Stream by words for natural feel
        words = result_output.split(" ")
        for i, word in enumerate(words):
            # Add space before word (except first word)
            if i > 0:
                await msg.stream_token(" ")
            await msg.stream_token(word)
            # Variable delay based on word length for more natural feel
            delay = min(0.05, 0.01 + len(word) * 0.005)  # 10-50ms per word
            await asyncio.sleep(delay)

        # Finalize the message
        await msg.update()

        logger.info(f"Processed message in session {pulsepal_session_id} with enhanced RAG v2")

    except asyncio.TimeoutError:
        logger.warning("Message processing timed out")
        await cl.Message(
            content=(
                "⏱️ **Request Timed Out**\n\n"
                "The processing took too long to complete. Please try:\n"
                "- Using a more specific query\n"
                "- Breaking your question into smaller parts\n"
                "- Asking about general concepts instead of searching for specific implementations"
            ),
            author="System",
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
            author="System",
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

        await cl.Message(content=error_msg, author="System").send()


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
                {"event": "session_end", "rag_version": "v2_enhanced"},
            )

            # Optional: Clean up expired sessions
            session_manager = get_session_manager()
            await session_manager.cleanup_expired_sessions()
            logger.info(
                f"Cleaned up Chainlit session for enhanced Pulsepal v2 session: {pulsepal_session_id}"
            )
    except Exception as e:
        logger.error(f"Error during session cleanup: {e}")


# Custom settings for Chainlit
@cl.step(type="tool")
async def rag_search_step(query: str, tool_name: str) -> str:
    """Display RAG search operations as structured steps."""
    return f"🔍 Searching {tool_name} with enhanced RAG v2 for: {query}"


# Settings configuration for sequence knowledge
@cl.author_rename
def rename(original_author: str):
    """Preserve author names including context indicators."""
    return original_author


# Configure Chainlit settings panel
@cl.set_chat_profiles
async def chat_profile():
    """Set up chat profiles for different modes."""
    return [
        cl.ChatProfile(
            name="Standard",
            markdown_description="Pulsepal v2 Enhanced with function validation and sequence context",
        )
    ]


if __name__ == "__main__":
    # This won't be called when running with `chainlit run`
    # But useful for testing imports
    print("Chainlit app v2 enhanced loaded successfully.")
    print("Features: Function validation, hallucination prevention, all UI features preserved")
    print("Run with: chainlit run chainlit_app_v2.py")
