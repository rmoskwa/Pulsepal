#!/usr/bin/env python3
"""
Chainlit web interface for Pulsepal with modern RAG service v2.

This version uses the simplified RAG service with enhanced function validation
to prevent hallucinations while maintaining clean architecture.
"""

import asyncio
import logging
import os

# Import existing Pulsepal components
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import chainlit as cl

sys.path.insert(0, str(Path(__file__).parent))

# Import components with function validation
from pulsepal.conversation_logger import get_conversation_logger
from pulsepal.dependencies import get_session_manager
from pulsepal.main_agent import create_pulsepal_session, pulsepal_agent
from pulsepal.semantic_router import SemanticRouter
from pulsepal.settings import get_settings
from pulsepal.startup import initialize_all_services

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
AUTH_ENABLED = True
try:
    logger.info("Attempting to import auth module...")
    from pulsepal.auth import API_KEYS, check_rate_limit, validate_api_key

    logger.info(
        f"Auth module imported successfully. API_KEYS type: {type(API_KEYS)}, length: {len(API_KEYS)}",
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
                    f"✅ Successful login for: {username or user_info['email']}",
                )
                return cl.User(
                    identifier=username or user_info["email"],
                    metadata={
                        "api_key": password,
                        "name": user_info["name"],
                        "limit": user_info.get("limit", 100),
                    },
                )
            logger.warning(f"❌ Failed login attempt with username: {username}")
            return None

        logger.info("Auth callback registered successfully")
    else:
        logger.info("🔓 Authentication disabled - no valid API keys configured")
        logger.info(
            f"Reason: len(API_KEYS)={len(API_KEYS)}, has only test-key={len(API_KEYS) == 1 and 'test-key' in API_KEYS}",
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

# Initialize all services at startup
# This will load models in the background
initialize_all_services()

# Global settings
settings = get_settings()

# Initialize conversation logger for debugging
conversation_logger = get_conversation_logger()

# Semantic router is already initialized in initialize_all_services()
# Just get the singleton instance (it won't reinitialize due to singleton pattern)
logger.info("Getting semantic router instance...")

_semantic_router = SemanticRouter()  # Will return existing singleton
logger.info("✅ Semantic router ready")

# Lock to prevent concurrent session initialization
# Will be initialized on first use to avoid event loop issues
_session_init_lock = None

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


@cl.on_stop
async def on_stop():
    """Handle session stop/disconnect gracefully."""
    pulsepal_session_id = cl.user_session.get("pulsepal_session_id")
    if pulsepal_session_id:
        logger.info(f"Session stopping/disconnecting: {pulsepal_session_id}")
        # Don't clear the session - keep it for potential reconnection
        # The session will timeout naturally after MAX_SESSION_DURATION_HOURS


@cl.on_chat_resume
async def on_chat_resume(thread):
    """Resume an existing chat session after reconnection.

    Note: Chainlit's session persistence is limited. This handler attempts to
    maintain continuity but may not always have access to previous session data.
    """
    logger.info("=" * 60)
    logger.info("CHAT RESUME TRIGGERED")
    logger.info("=" * 60)

    # Since Chainlit doesn't provide reliable thread persistence,
    # we'll try to check if there's an active session in our session manager
    session_manager = get_session_manager()

    # Get the most recent active session (if any)
    # This is a workaround since Chainlit doesn't give us the session ID directly
    active_sessions = session_manager.get_active_sessions()

    if active_sessions:
        # Use the most recent session
        pulsepal_session_id = active_sessions[-1]
        deps = session_manager.get_session(pulsepal_session_id)

        if deps:
            logger.info(f"Found active session: {pulsepal_session_id}")
            cl.user_session.set("pulsepal_session_id", pulsepal_session_id)
            cl.user_session.set("pulsepal_deps", deps)

            # Restore the chat settings
            if deps.conversation_context:
                settings = [
                    cl.input_widget.TextInput(
                        id="sequence_knowledge",
                        label="🎯 Sequence Knowledge",
                        description="Add your sequence-specific context.",
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

                await cl.Message(
                    content=f"✅ **Session resumed!** Continuing conversation with {deps.conversation_context.conversation_count} previous messages.",
                    author="system",
                ).send()

                logger.info(
                    f"✅ Restored session context with {deps.conversation_context.conversation_count} messages"
                )
                return

    # If no active session found, start fresh
    logger.info("No active session found, starting new one")
    await start()


@cl.on_chat_start
async def start():
    """Initialize chat session with Pulsepal agent using enhanced RAG v2."""
    global _session_init_lock
    if _session_init_lock is None:
        _session_init_lock = asyncio.Lock()

    async with _session_init_lock:
        try:
            # Check if session already exists to prevent duplicate initialization
            existing_session_id = cl.user_session.get("pulsepal_session_id")
            if existing_session_id:
                logger.warning(
                    f"Session already exists: {existing_session_id}, skipping duplicate initialization"
                )
                return

            # Create new Pulsepal session with v2 enhanced components
            session_id = str(uuid.uuid4())
            pulsepal_session_id, deps = await create_pulsepal_session(session_id)

            # Store session info in Chainlit user session
            cl.user_session.set("pulsepal_session_id", pulsepal_session_id)
            cl.user_session.set("pulsepal_deps", deps)

            # Store session ID for potential recovery
            # Note: Chainlit's thread persistence is limited, so we primarily rely on
            # the session manager's in-memory storage for session continuity
            logger.info(f"Session initialized with ID: {pulsepal_session_id}")

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
                        f"User metadata: {user.metadata if hasattr(user, 'metadata') else 'No metadata'}",
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
            welcome_msg = f"""🧠 **Welcome to Pulsepal - Intelligent MRI Assistant with Pulseq Expert!**{auth_info}{context_status}



I'm an advanced AI with comprehensive MRI physics and Pulseq knowledge,

**💡 My Capabilities:**
- Answer questions about MRI physics and Pulseq sequences
- Provide details about Pulseq functions and parameters
- Assist with sequence design and optimization
- Troubleshoot common pulse sequence design and MRI issues
- And more! Just ask me anything related to MRI sequences or Pulseq.

**NOTE**: Pulsepal is currently not designed to be a coder! But it does ok. Double check all code that Pulsepal independently produces!! Or just tell Pulsepal to correct it :)

What would you like to explore about MRI sequence programming today?"""

            # Send welcome message
            await cl.Message(content=welcome_msg).send()

            logger.info(
                f"Started Chainlit session with enhanced Pulsepal v2 session: {pulsepal_session_id}",
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
        # Debug: Log message structure
        logger.info(f"Message received. Has elements: {bool(message.elements)}")
        if message.elements:
            logger.info(f"Number of elements: {len(message.elements)}")
            for i, elem in enumerate(message.elements):
                logger.info(
                    f"Element {i}: type={type(elem).__name__}, name={getattr(elem, 'name', 'N/A')}, mime={getattr(elem, 'mime', 'N/A')}"
                )

        # Extract code from uploaded .m files only
        code_context = ""
        uploaded_files_info = []

        if message.elements:
            for element in message.elements:
                # Check element attributes for debugging
                file_name = getattr(element, "name", None)
                file_mime = getattr(element, "mime", None)
                file_path = getattr(element, "path", None)

                # Debug logging
                logger.info(f"Processing element: type={type(element).__name__}")
                logger.info(f"  - name: {file_name}")
                logger.info(f"  - mime: {file_mime}")
                logger.info(f"  - path: {file_path}")

                # Only accept .m files - check both name extension and MIME type
                if file_name:
                    file_ext = os.path.splitext(file_name)[1].lower()

                    # Only process .m (MATLAB) files
                    if file_ext == ".m":
                        try:
                            # Get file content - handle different element types
                            file_content = None

                            # Try different methods to extract content based on Chainlit version differences

                            # Method 1: Try path first (newer Chainlit versions store files in .files directory)
                            if file_path:
                                logger.info(
                                    f"Attempting to read from path: {file_path}"
                                )
                                paths_to_try = [
                                    file_path,  # Direct path
                                    os.path.join(
                                        ".files", file_path
                                    ),  # Relative to .files
                                    os.path.join(
                                        os.getcwd(), file_path
                                    ),  # Absolute from cwd
                                    os.path.join(
                                        os.getcwd(), ".files", file_path
                                    ),  # Absolute .files path
                                ]

                                for try_path in paths_to_try:
                                    if os.path.exists(try_path):
                                        logger.info(f"Found file at: {try_path}")
                                        try:
                                            with open(
                                                try_path, "r", encoding="utf-8"
                                            ) as f:
                                                file_content = f.read()
                                                logger.info(
                                                    f"Successfully read {len(file_content)} bytes from {try_path}"
                                                )
                                                break
                                        except Exception as e:
                                            logger.error(
                                                f"Error reading from {try_path}: {e}"
                                            )

                                if file_content is None:
                                    logger.warning(
                                        "Could not find file at any of the tried paths"
                                    )

                            # Method 2: Try content attribute (older versions or AskFileMessage)
                            if file_content is None and hasattr(element, "content"):
                                logger.info("Attempting to read from content attribute")
                                if element.content is not None:
                                    if isinstance(element.content, bytes):
                                        file_content = element.content.decode("utf-8")
                                        logger.info(
                                            f"Successfully decoded {len(file_content)} bytes from content"
                                        )
                                    elif isinstance(element.content, str):
                                        file_content = element.content
                                        logger.info(
                                            f"Successfully got {len(file_content)} chars from content string"
                                        )

                            # Method 3: Check if it's a cl.File object with specific attributes
                            if file_content is None:
                                # Log what we have for debugging
                                logger.info(
                                    f"Still no content. Element class: {element.__class__.__name__}"
                                )
                                logger.info(
                                    f"Element dict (if available): {getattr(element, '__dict__', 'N/A')}"
                                )

                            if file_content is None:
                                logger.warning(
                                    f"Could not extract content from file: {file_name}. "
                                    f"Element type: {type(element).__name__}, "
                                    f"Has content: {hasattr(element, 'content')}, "
                                    f"Has path: {hasattr(element, 'path')}, "
                                    f"Path value: {getattr(element, 'path', 'N/A')}"
                                )
                                await cl.Message(
                                    content=f"⚠️ Unable to read file '{file_name}'. Please try uploading again.",
                                    author="System",
                                ).send()
                                continue

                            # Check file size (limit to 1MB for performance)
                            file_size = len(file_content)
                            if file_size > 1024 * 1024:
                                await cl.Message(
                                    content=f"⚠️ File '{file_name}' is too large ({file_size/1024:.1f}KB > 1MB limit). Please upload a smaller file.",
                                    author="System",
                                ).send()
                                continue

                            # Add file content with XML tags
                            code_context += f"\n\n<user-provided-file: {file_name}>\n{file_content}\n</user-provided-file: {file_name}>\n"

                            # Track uploaded file info
                            lines = file_content.split("\n")
                            uploaded_files_info.append(
                                {
                                    "name": file_name,
                                    "size": file_size,
                                    "lines": len(lines),
                                }
                            )

                            # Send confirmation to user
                            await cl.Message(
                                content=f"✅ Loaded MATLAB file: `{file_name}` ({file_size} bytes, {len(lines)} lines)",
                                author="System",
                            ).send()

                            logger.info(
                                f"Successfully loaded MATLAB file: {file_name} ({file_size} bytes)"
                            )

                        except UnicodeDecodeError as e:
                            logger.error(
                                f"Unicode decode error for file {file_name}: {e}"
                            )
                            await cl.Message(
                                content=f"❌ File '{file_name}' contains invalid characters. Please ensure it's a valid text file.",
                                author="System",
                            ).send()
                        except Exception as e:
                            logger.error(f"Error loading file {file_name}: {e}")
                            await cl.Message(
                                content=f"❌ Error loading file '{file_name}': {str(e)}",
                                author="System",
                            ).send()
                    else:
                        # Unsupported file type - only .m files are accepted
                        await cl.Message(
                            content=f"⚠️ Only .m (MATLAB) files are accepted. File '{file_name}' has extension '{file_ext}' and was not processed.",
                            author="System",
                        ).send()

        # Combine user query with code context
        enhanced_query = message.content
        if code_context:
            enhanced_query = f"{message.content}\n{code_context}"
            logger.info(
                f"Enhanced query with {len(code_context)} bytes of code from {len(uploaded_files_info)} MATLAB file(s)"
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
                        f"Rate limit exceeded for user: {user_name} (API key: {api_key[:8]}...)",
                    )
                    await cl.Message(
                        content=f"⚠️ **Rate limit exceeded**\n\nHi {user_name}, you've reached your limit of {limit} requests per hour.\n\nPlease wait a few minutes before sending more requests.",
                        author="System",
                    ).send()
                    return
                if api_key:
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
            f"Session {pulsepal_session_id} - History length: {len(deps.conversation_context.conversation_history)}",
        )
        if deps.conversation_context.conversation_history:
            last_entry = deps.conversation_context.conversation_history[-1]
            logger.debug(
                f"Session {pulsepal_session_id} - Last message: {last_entry['role']}: {last_entry['content'][:50]}...",
            )

        # Apply semantic routing to determine if RAG is needed
        from pulsepal.main_agent import apply_semantic_routing

        apply_semantic_routing(enhanced_query, deps)

        # Show typing indicator with intelligent status
        async with cl.Step(
            name="🧠 Processing with Enhanced RAG v2 (Function Validation Active)...",
        ) as step:
            try:
                # Add user message to conversation context (use enhanced_query if code was uploaded)
                deps.conversation_context.add_conversation("user", enhanced_query)

                # Log user message for debugging
                conversation_logger.log_conversation(
                    pulsepal_session_id,
                    "user",
                    enhanced_query,
                    {"rag_version": "v2_enhanced"},
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
                        f"Including sequence context: {len(sequence_context)} chars",
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
                    conversation_logger,
                )

                # Store routing decision in deps
                deps.routing_decision = routing_decision

                # Apply routing decision but don't inject detected functions
                # Only use for logging and validation errors
                if routing_decision.detected_functions:
                    deps.validation_errors = routing_decision.validation_errors

                    logger.info(
                        f"Function detector found {len(routing_decision.detected_functions)} function(s): "
                        f"{[f['name'] for f in routing_decision.detected_functions]}",
                    )

                    if routing_decision.validation_errors:
                        logger.warning(
                            f"Validation errors detected: {routing_decision.validation_errors}"
                        )

                # Check if RAG search should be forced based on routing decision
                from pulsepal.semantic_router import QueryRoute

                if routing_decision.route == QueryRoute.FORCE_RAG:
                    logger.info(
                        f"Semantic router recommends RAG search (confidence: {routing_decision.confidence:.2f})"
                    )
                    # Set the force_rag flag on deps so it's available to agent
                    deps.force_rag = True
                    # Inject the hint into the query for this message only
                    query_for_agent = (
                        f"{query_with_context}\n\n⚠️ **IMPORTANT: You MUST use the search_pulseq_knowledge tool to answer this question accurately. "
                        f"The query contains Pulseq-specific functions that require searching the knowledge base. "
                        f"DO NOT rely on your training data - search the database first, then provide your answer based on the search results.**"
                    )
                    logger.info(
                        "💡 Injecting strong RAG search directive into Chainlit query"
                    )
                else:
                    deps.force_rag = False
                    query_for_agent = query_with_context

                # Log the detection but don't restrict Gemini's choices
                logger.debug(
                    f"Function detection complete. Route: {routing_decision.route.value}"
                )

                # Run agent with potentially modified query and model settings
                result = await pulsepal_agent.run(
                    query_for_agent, deps=deps, model_settings={"temperature": 1.0}
                )

                # Add response to conversation history
                # Use result.output for modern pydantic-ai
                response_text = (
                    result.output if hasattr(result, "output") else str(result)
                )
                deps.conversation_context.add_conversation("assistant", response_text)

                # Log assistant response for debugging
                conversation_logger.log_conversation(
                    pulsepal_session_id,
                    "assistant",
                    response_text,
                    {"rag_version": "v2_enhanced"},
                )

                step.output = "✅ Response ready (Enhanced RAG v2 with validation)"

            except Exception as e:
                logger.error(f"Error running Pulsepal agent: {e}")
                step.output = f"❌ Error: {e}"
                result_output = f"I apologize, but I encountered an error: {e}\n\nPlease try rephrasing your question or check that all services are running properly."
            else:
                result_output = response_text

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

        logger.info(
            f"Processed message in session {pulsepal_session_id} with enhanced RAG v2",
        )

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
            # Check if session actually exists before trying to clean it up
            session_manager = get_session_manager()

            # Only log and clean if session exists
            if pulsepal_session_id in session_manager.sessions:
                # Log session end for debugging
                conversation_logger.log_conversation(
                    pulsepal_session_id,
                    "system",
                    "Session ended",
                    {"event": "session_end", "rag_version": "v2_enhanced"},
                )

                # Clean up this specific session
                session_manager.cleanup_session(pulsepal_session_id)
                logger.info(
                    f"Cleaned up Chainlit session for enhanced Pulsepal v2 session: {pulsepal_session_id}",
                )
            else:
                # Session already cleaned up, skip logging to avoid spam
                logger.debug(
                    f"Session {pulsepal_session_id} already cleaned up, skipping"
                )

            # Clean up other expired sessions (but not too frequently)
            # Only do this occasionally to avoid excessive cleanup attempts
            import random

            if random.random() < 0.1:  # 10% chance to run cleanup
                await session_manager.cleanup_expired_sessions()
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
        ),
    ]


if __name__ == "__main__":
    # This won't be called when running with `chainlit run`
    # But useful for testing imports
    print("Chainlit app v2 enhanced loaded successfully.")
    print(
        "Features: Function validation, hallucination prevention, all UI features preserved",
    )
    print("Run with: chainlit run chainlit_app_v2.py")
