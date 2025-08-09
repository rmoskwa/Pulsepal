#!/usr/bin/env python3
"""
Chainlit web interface for Pulsepal PydanticAI multi-agent system.

Provides a modern web UI that directly integrates with the existing PydanticAI agents,
session management, and RAG capabilities without requiring API changes.
"""

import asyncio
import logging
import uuid
from typing import Optional, Dict, Any
import os
from datetime import datetime

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
if "ALPHA_API_KEYS" in os.environ:
    # Don't log the actual keys for security
    logger.info(f"ALPHA_API_KEYS length: {len(os.environ['ALPHA_API_KEYS'])}")
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
logger.info("=" * 60)

# Global settings
settings = get_settings()

# Initialize conversation logger for debugging
conversation_logger = get_conversation_logger()

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
            pulsepal_session_id, "system", "Session started", {"event": "session_start"}
        )

        # Configure settings panel for sequence knowledge
        # Add template hint in description
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

        # Get supported languages for welcome message
        lang_list = ", ".join(
            [lang.upper() for lang in sorted(SUPPORTED_LANGUAGES.keys())]
        )

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

        # Note: Action buttons disabled for now - Settings panel provides all functionality
        # Check if context is already active
        context_status = ""
        if (
            deps.conversation_context.use_sequence_context
            and deps.conversation_context.sequence_knowledge
        ):
            context_status = "\n\n🎯 **[Sequence Context Active]** - Your sequence-specific knowledge is being used."

        # Send welcome message
        welcome_msg = f"""🧠 **Welcome to Pulsepal - Your Intelligent MRI Programming Assistant!**{auth_info}{context_status}

I'm an advanced AI with comprehensive knowledge of MRI physics and Pulseq programming, enhanced with access to specialized documentation when needed.

**🚀 What's New:**
- **🎯 Sequence Knowledge**: Add your sequence context via Settings (gear icon) for targeted assistance
  - Open Settings and add your sequence details (e.g., "EPI sequence for fMRI at 3T")
  - Enable "Sequence Context" toggle to activate
  - Your context will be considered in all responses!
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

        # Send welcome message without actions (they're temporarily disabled)
        await cl.Message(content=welcome_msg).send()

        logger.info(
            f"Started Chainlit session with Pulsepal session: {pulsepal_session_id}"
        )

    except Exception as e:
        logger.error(f"Failed to initialize chat session: {e}")
        await cl.Message(
            content=f"❌ **Error**: Failed to initialize Pulsepal session: {e}\n\nPlease refresh the page and try again.",
            author="System",
        ).send()


@cl.action_callback("show_template")
async def show_template_action(action: cl.Action):
    """Display the sequence knowledge template."""
    template_msg = f"""📝 **Sequence Knowledge Template**

Here's a comprehensive template to help you structure your sequence-specific knowledge:

```markdown
{SEQUENCE_KNOWLEDGE_TEMPLATE}
```

**How to Use:**
1. Copy the template above
2. Fill in your sequence-specific details  
3. Use the settings panel (⚙️ icon) to paste and enable the context
4. Your sequence knowledge will then inform all my responses!

💡 **Tip**: You can customize any section based on your specific needs. The template is designed to capture the most important sequence parameters and implementation details."""

    await cl.Message(content=template_msg).send()


@cl.action_callback("toggle_context")
async def toggle_context_action(action: cl.Action):
    """Toggle sequence context on/off."""
    deps = cl.user_session.get("pulsepal_deps")
    if not deps:
        await cl.Message(content="❌ Session error. Please refresh the page.").send()
        return

    # Toggle the context
    deps.conversation_context.use_sequence_context = (
        not deps.conversation_context.use_sequence_context
    )

    if deps.conversation_context.use_sequence_context:
        if deps.conversation_context.sequence_knowledge:
            status_msg = "🎯 **Sequence Context Enabled**\n\nYour sequence knowledge is now active and will inform all responses. You'll see the [Sequence Context Active] indicator in my messages."
        else:
            status_msg = "⚠️ **Context Enabled but Empty**\n\nSequence context is enabled, but no sequence knowledge has been added yet. Use the settings panel (⚙️) to add your sequence details."
    else:
        status_msg = "⏸️ **Sequence Context Disabled**\n\nSequence context has been turned off. I'll provide general responses without your sequence-specific knowledge."

    await cl.Message(content=status_msg).send()


@cl.action_callback("download_context")
async def download_context_action(action: cl.Action):
    """Export sequence knowledge as downloadable markdown."""
    deps = cl.user_session.get("pulsepal_deps")
    if not deps:
        await cl.Message(content="❌ Session error. Please refresh the page.").send()
        return

    if not deps.conversation_context.sequence_knowledge:
        await cl.Message(
            content="📄 **No Sequence Knowledge to Export**\n\nYou haven't added any sequence knowledge yet. Use the settings panel (⚙️) to add sequence details, then you can export them."
        ).send()
        return

    # Generate export content
    export_content = deps.conversation_context.export_sequence_knowledge()

    # Create downloadable file
    filename = f"sequence_knowledge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    # Create file element for download
    file_element = cl.File(
        name=filename, content=export_content.encode("utf-8"), display="side"
    )

    download_msg = f"""💾 **Sequence Knowledge Exported**

Your sequence knowledge has been exported as a markdown file: `{filename}`

**Export Details:**
- Session ID: {deps.conversation_context.session_id}
- Export Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Context Status: {"Enabled" if deps.conversation_context.use_sequence_context else "Disabled"}
- Content Length: {len(deps.conversation_context.sequence_knowledge)} characters

📄 The file contains your sequence knowledge with session metadata for future reference."""

    await cl.Message(content=download_msg, elements=[file_element]).send()


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


# Action callbacks temporarily disabled - will be re-enabled once Chainlit Action API is clarified
# @cl.action_callback("show_template")
# async def show_template(action: cl.Action):
#     """Show the sequence knowledge template."""
#     template_msg = f"""📋 **Sequence Knowledge Template**
#
# Use this template to provide context about your MRI sequence. Copy and modify as needed:
#
# ```markdown
# {SEQUENCE_KNOWLEDGE_TEMPLATE}
# ```
#
# **How to use:**
# 1. Copy this template
# 2. Open Settings (gear icon)
# 3. Paste into "Sequence Knowledge" field
# 4. Fill in your specific details
# 5. Enable "Sequence Context" toggle
# 6. Click Save
#
# Your sequence context will then be considered in all responses!
# """
#     await cl.Message(content=template_msg).send()
#
#
# @cl.action_callback("toggle_context")
# async def toggle_context(action: cl.Action):
#     """Toggle sequence context on/off."""
#     deps = cl.user_session.get("pulsepal_deps")
#     if not deps:
#         await cl.Message(content="❌ Session error. Please refresh the page.").send()
#         return
#
#     # Toggle the setting
#     deps.conversation_context.use_sequence_context = not deps.conversation_context.use_sequence_context
#     new_state = deps.conversation_context.use_sequence_context
#
#     if new_state:
#         if deps.conversation_context.sequence_knowledge:
#             msg = "✅ **Sequence Context Enabled**\n\nYour sequence knowledge will now be considered in all responses."
#         else:
#             msg = "⚠️ **Sequence Context Enabled** (but empty)\n\nAdd sequence knowledge in Settings to get targeted assistance."
#     else:
#         msg = "🔄 **Sequence Context Disabled**\n\nResponses will be general without sequence-specific context."
#
#     await cl.Message(content=msg).send()
#
#
# @cl.action_callback("download_context")
# async def download_context(action: cl.Action):
#     """Download sequence knowledge as markdown."""
#     deps = cl.user_session.get("pulsepal_deps")
#     if not deps:
#         await cl.Message(content="❌ Session error. Please refresh the page.").send()
#         return
#
#     if not deps.conversation_context.sequence_knowledge:
#         await cl.Message(content="⚠️ No sequence knowledge to download. Add some in Settings first!").send()
#         return
#
#     # Generate export content
#     export_content = deps.conversation_context.export_sequence_knowledge()
#
#     # Create filename with timestamp
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"sequence_knowledge_{timestamp}.md"
#
#     # Create file element
#     file_element = cl.File(
#         name=filename,
#         content=export_content.encode('utf-8'),
#         display="inline"
#     )
#
#     await cl.Message(
#         content=f"📥 **Sequence Knowledge Exported**\n\nDownload your sequence context as: `{filename}`",
#         elements=[file_element]
#     ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages and respond using Pulsepal agent."""
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
🔒 **Authentication**: Enabled"""
            else:
                info_msg = "🔓 **Authentication**: Disabled (local mode)"

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
        async with cl.Step(name="🧠 Analyzing your query...") as step:
            try:
                # Add user message to conversation context (use enhanced_query if code was uploaded)
                deps.conversation_context.add_conversation("user", enhanced_query)

                # Log user message for debugging
                conversation_logger.log_conversation(
                    pulsepal_session_id, "user", enhanced_query
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

                # Run agent with query including context
                result = await pulsepal_agent.run(query_with_context, deps=deps)

                # Add response to conversation history
                deps.conversation_context.add_conversation("assistant", result.data)

                # Log assistant response for debugging
                conversation_logger.log_conversation(
                    pulsepal_session_id, "assistant", result.data
                )

                step.output = "✅ Response ready"

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
        msg = cl.Message(content="", author=f"{context_prefix}Pulsepal")

        # Option 1: Stream by words (more natural)
        words = result_output.split(" ")
        for i, word in enumerate(words):
            # Add space before word (except first word)
            if i > 0:
                await msg.stream_token(" ")
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
                pulsepal_session_id, "system", "Session ended", {"event": "session_end"}
            )

            # Optional: Clean up expired sessions
            session_manager = get_session_manager()
            await session_manager.cleanup_expired_sessions()
            logger.info(
                f"Cleaned up Chainlit session for Pulsepal session: {pulsepal_session_id}"
            )
    except Exception as e:
        logger.error(f"Error during session cleanup: {e}")


# Custom settings for Chainlit
@cl.step(type="tool")
async def rag_search_step(query: str, tool_name: str) -> str:
    """Display RAG search operations as structured steps."""
    return f"🔍 Searching {tool_name} for: {query}"


# Note: cl.on_error is not available in Chainlit 2.6.3
# Error handling is done within individual handlers


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
            markdown_description="Standard Pulsepal mode with optional sequence context",
        )
    ]


if __name__ == "__main__":
    # This won't be called when running with `chainlit run`
    # But useful for testing imports
    print("Chainlit app loaded successfully. Run with: chainlit run chainlit_app.py")
