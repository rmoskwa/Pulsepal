# PulsePal API Reference

## Core Modules

### pulsepal.main_agent

The main agent module contains the core PulsePal intelligent agent.

#### Functions

##### `run_pulsepal(query: str, session_id: str = None) -> tuple[str, str]`

Run PulsePal with a query and optional session management.

**Parameters:**
- `query` (str): The user's question or request about Pulseq/MRI
- `session_id` (str, optional): Existing session ID to continue conversation

**Returns:**
- `tuple[str, str]`: Session ID and response text

**Example:**
```python
from pulsepal.main_agent import run_pulsepal

session_id, response = await run_pulsepal(
    "How do I create a gradient echo sequence?",
    session_id="existing-session-123"
)
```

##### `pulsepal_agent`

The main PydanticAI agent instance configured with Google Gemini.

**Usage:**
```python
from pulsepal.main_agent import pulsepal_agent

result = await pulsepal_agent.run(
    "Your query here",
    deps=dependencies
)
```

---

### pulsepal.rag_service

RAG (Retrieval Augmented Generation) service for Pulseq documentation search.

#### Classes

##### `ModernPulseqRAG`

Advanced RAG service with hybrid search capabilities.

**Methods:**

###### `search_pulseq_knowledge(query: str, search_type: str = "auto") -> str`

Search Pulseq documentation and return relevant information.

**Parameters:**
- `query` (str): Search query
- `search_type` (str): One of "auto", "documentation", "functions", "examples", "code"

**Returns:**
- `str`: Formatted search results

**Example:**
```python
from pulsepal.rag_service import ModernPulseqRAG

rag = ModernPulseqRAG()
results = await rag.search_pulseq_knowledge(
    "gradient echo parameters",
    search_type="documentation"
)
```

##### `SupabaseRAGClient`

Low-level client for Supabase vector database operations.

**Methods:**

###### `hybrid_search(query: str, limit: int = 5) -> List[Dict]`

Perform hybrid vector + keyword search.

**Parameters:**
- `query` (str): Search query
- `limit` (int): Maximum results to return

**Returns:**
- `List[Dict]`: Search results with metadata

---

### pulsepal.tools

Tool functions for the PulsePal agent.

#### Functions

##### `search_pulseq_knowledge(ctx: RunContext, query: str, search_type: str = "auto") -> str`

Agent tool for searching Pulseq knowledge base.

**Parameters:**
- `ctx`: PydanticAI run context
- `query` (str): Search query
- `search_type` (str): Search method selection

**Returns:**
- `str`: Formatted search results for agent use

**Example:**
```python
from pulsepal.tools import search_pulseq_knowledge

# Used internally by the agent
@pulsepal_agent.tool
async def search_tool(ctx, query):
    return await search_pulseq_knowledge(ctx, query)
```

---

### pulsepal.dependencies

Session management and conversation context.

#### Classes

##### `SessionManager`

Manages user sessions and conversation history.

**Methods:**

###### `create_session() -> str`

Create a new session with unique ID.

**Returns:**
- `str`: New session ID

###### `get_session(session_id: str) -> ConversationContext`

Retrieve existing session context.

**Parameters:**
- `session_id` (str): Session identifier

**Returns:**
- `ConversationContext`: Session data

###### `cleanup_old_sessions(max_age_hours: int = 24)`

Remove expired sessions.

**Parameters:**
- `max_age_hours` (int): Maximum session age

##### `ConversationContext`

Dataclass for storing conversation state.

**Attributes:**
- `session_id` (str): Unique session identifier
- `conversation_history` (List[Dict]): Previous messages
- `preferred_language` (str): "matlab" or "python"
- `code_examples` (List[Dict]): Stored code snippets
- `session_start_time` (datetime): Creation timestamp
- `last_activity` (datetime): Last interaction time

**Example:**
```python
from pulsepal.dependencies import SessionManager, ConversationContext

manager = SessionManager()
session_id = manager.create_session()
context = manager.get_session(session_id)

# Access conversation history
for message in context.conversation_history:
    print(f"{message['role']}: {message['content']}")
```

##### `PulsePalDependencies`

Dependencies container for the agent.

**Attributes:**
- `conversation_context` (ConversationContext): Current session
- `session_manager` (SessionManager): Session manager instance
- `rag_initialized` (bool): RAG service status
- `force_rag` (bool): Force RAG search
- `skip_rag` (bool): Skip RAG search

---

### pulsepal.settings

Configuration management using pydantic-settings.

#### Classes

##### `Settings`

Application settings with environment variable support.

**Attributes:**
- `google_api_key` (str): Google API key for Gemini
- `llm_model` (str): Model name (default: "gemini-2.5-flash")
- `supabase_url` (str): Supabase project URL
- `supabase_key` (str): Supabase API key
- `max_session_duration_hours` (int): Session timeout (default: 24)
- `max_conversation_history` (int): Max messages per session (default: 100)
- `use_hybrid_search` (bool): Enable hybrid search (default: True)

#### Functions

##### `get_settings() -> Settings`

Load settings from environment variables.

**Returns:**
- `Settings`: Configured settings instance

**Example:**
```python
from pulsepal.settings import get_settings

settings = get_settings()
print(f"Using model: {settings.llm_model}")
print(f"Session timeout: {settings.max_session_duration_hours} hours")
```

##### `get_llm_model() -> GeminiModel`

Get configured Gemini model instance.

**Returns:**
- `GeminiModel`: Configured model for PydanticAI

---

## CLI Interface

### run_pulsepal.py

Command-line interface for PulsePal.

#### Usage

```bash
# Interactive mode
python run_pulsepal.py

# Single query
python run_pulsepal.py "Your question here"

# With session
python run_pulsepal.py --session-id <id>

# Help
python run_pulsepal.py --help
```

#### Arguments

- `query` (positional, optional): Direct query to process
- `--session-id`: Continue existing session
- `--interactive`: Force interactive mode
- `--help`: Show help message

---

## Web Interface

### chainlit_app.py

Chainlit web interface for PulsePal.

#### Starting the Server

```bash
chainlit run chainlit_app.py
```

#### Features

- Real-time streaming responses
- Session persistence
- Markdown rendering
- Code syntax highlighting
- File upload support

#### Configuration

Configure in `.chainlit/config.toml`:

```toml
[project]
name = "PulsePal"
description = "MRI Sequence Assistant"

[UI]
theme = "light"
```

---

## Error Handling

### Common Exceptions

#### `SessionNotFoundError`

Raised when session ID doesn't exist.

```python
try:
    context = manager.get_session("invalid-id")
except SessionNotFoundError:
    context = manager.create_session()
```

#### `RAGServiceError`

Raised when RAG search fails.

```python
try:
    results = await rag.search_pulseq_knowledge(query)
except RAGServiceError as e:
    # Fallback to agent's built-in knowledge
    results = None
```

#### `ConfigurationError`

Raised when required environment variables are missing.

```python
try:
    settings = get_settings()
except ConfigurationError as e:
    print(f"Missing configuration: {e}")
```

---

## Environment Variables

Required environment variables in `.env`:

```bash
# Google Gemini API
GOOGLE_API_KEY=your-api-key
LLM_MODEL=gemini-2.5-flash

# Supabase Vector Database
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-key

# Optional: Google Embeddings
GOOGLE_API_KEY_EMBEDDING=your-embedding-key

# Session Configuration
MAX_SESSION_DURATION_HOURS=24
MAX_CONVERSATION_HISTORY=100

# Search Configuration
USE_HYBRID_SEARCH=true
```

---

## Return Types

### Search Results Format

```python
{
    "results": [
        {
            "content": "Documentation text",
            "metadata": {
                "source": "file.md",
                "section": "gradient_echo",
                "relevance": 0.95
            }
        }
    ],
    "query": "original query",
    "search_type": "documentation"
}
```

### Session Format

```python
{
    "session_id": "uuid-string",
    "conversation_history": [
        {
            "role": "user",
            "content": "question",
            "timestamp": "2025-01-18T10:00:00"
        },
        {
            "role": "assistant",
            "content": "response",
            "timestamp": "2025-01-18T10:00:01"
        }
    ],
    "preferred_language": "matlab",
    "created_at": "2025-01-18T09:59:00"
}
```

---

## Examples

### Basic Usage

```python
import asyncio
from pulsepal.main_agent import run_pulsepal

async def main():
    # Simple query
    session_id, response = await run_pulsepal(
        "What is a gradient echo sequence?"
    )
    print(response)
    
    # Follow-up with session
    _, response2 = await run_pulsepal(
        "Can you show me the code?",
        session_id=session_id
    )
    print(response2)

asyncio.run(main())
```

### Custom RAG Search

```python
from pulsepal.rag_service import ModernPulseqRAG

async def search_examples():
    rag = ModernPulseqRAG()
    
    # Search for code examples
    examples = await rag.search_pulseq_knowledge(
        "EPI sequence implementation",
        search_type="examples"
    )
    
    # Search for function documentation
    functions = await rag.search_pulseq_knowledge(
        "makeArbitraryGrad parameters",
        search_type="functions"
    )
    
    return examples, functions
```

### Session Management

```python
from pulsepal.dependencies import SessionManager

# Create manager
manager = SessionManager()

# New session
session_id = manager.create_session()

# Get session
context = manager.get_session(session_id)

# Update context
context.preferred_language = "python"
context.conversation_history.append({
    "role": "user",
    "content": "Show me Python code"
})

# Save session
manager.save_session(context)

# Cleanup old sessions
manager.cleanup_old_sessions(max_age_hours=48)
```

---

## Testing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=pulsepal

# Specific module
pytest tests/unit/test_main_agent.py
```

### Test Structure

```
tests/
├── unit/          # Unit tests
├── integration/   # Integration tests
├── e2e/          # End-to-end tests
└── fixtures/     # Test data
```

---

## Support

For issues or questions:
- Email: rmoskwa@wisc.edu
- GitHub: https://github.com/rmoskwa/Pulsepal/issues
- Documentation: This file