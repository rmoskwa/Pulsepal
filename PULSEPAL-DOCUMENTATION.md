# PulsePal Documentation

**PulsePal** is a sophisticated PydanticAI-based multi-agent system designed specifically for MRI sequence programming with Pulseq. It combines advanced RAG (Retrieval Augmented Generation) capabilities, specialized physics expertise, and both web and CLI interfaces to provide comprehensive assistance for MRI researchers and programmers.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [Usage Guide](#usage-guide)
5. [Core Components](#core-components)
6. [Configuration](#configuration)
7. [API Reference](#api-reference)
8. [Testing](#testing)
9. [Deployment](#deployment)
10. [Development Guide](#development-guide)

## Overview

### What is PulsePal?

PulsePal is an AI-powered assistant that helps researchers and programmers work with Pulseq v1.5.0 across MATLAB, Octave, and Python environments. It provides:

- **Code Generation**: Creates MRI sequences in multiple programming languages
- **Documentation Search**: Advanced RAG system with comprehensive Pulseq documentation
- **Physics Expertise**: Specialized MRI physics explanations via agent delegation
- **Session Management**: Conversation continuity across interactions
- **Multi-Interface**: Both web UI (Chainlit) and command-line interfaces

### Key Features

🔬 **Multi-Agent Architecture**: Specialized agents for programming and physics
📚 **Advanced RAG System**: Search comprehensive Pulseq documentation and code examples
🌐 **Modern Web Interface**: Beautiful Chainlit-based UI with markdown rendering
💬 **Session Continuity**: Maintains context across conversations
🔧 **Multi-Language Support**: MATLAB (default), Python, and Octave
⚡ **Direct Integration**: No APIs - calls PydanticAI agents directly
🛡️ **Robust Error Handling**: Graceful fallback when services are unavailable

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        PulsePal System                      │
├─────────────────────────────────────────────────────────────┤
│  Interfaces:                                                │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Chainlit Web UI │    │   CLI Interface │                │
│  │  (chainlit_app) │    │ (run_pulsepal)  │                │
│  └─────────────────┘    └─────────────────┘                │
├─────────────────────────────────────────────────────────────┤
│  Core Agents:                                               │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │  Pulsepal Agent │    │ MRI Expert Agent│                │
│  │ (main_agent.py) │◄──►│(mri_expert.py)  │                │
│  └─────────────────┘    └─────────────────┘                │
├─────────────────────────────────────────────────────────────┤
│  Support Systems:                                           │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │  RAG Service    │    │Session Manager  │                │
│  │ (rag_service.py)│    │(dependencies.py)│                │
│  └─────────────────┘    └─────────────────┘                │
├─────────────────────────────────────────────────────────────┤
│  Data Layer:                                                │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │  Supabase DB    │    │  BGE Embeddings │                │
│  │ (vector store)  │    │   (local model) │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### Agent Architecture

**Pulsepal Agent** (`pulsepal/main_agent.py`)
- Main programming assistant
- Handles code generation and debugging
- Manages RAG searches and agent delegation
- Maintains conversation context

**MRI Expert Agent** (`pulsepal/mri_expert_agent.py`)  
- Specialized physics consultant
- Provides educational MRI theory explanations
- Handles k-space, RF pulse, and gradient theory
- Seamlessly integrated via tool delegation

### RAG System

The RAG (Retrieval Augmented Generation) system provides:
- **Vector Search**: BGE-large-en-v1.5 embeddings for semantic search
- **Hybrid Search**: Combines vector and keyword search for optimal results
- **Document Store**: Supabase vector database with comprehensive Pulseq documentation
- **Code Examples**: Searchable repository of MATLAB, Python, and Octave code

## Installation & Setup

### Prerequisites

- Python 3.8+
- Access to Google Gemini API (for LLM)
- Supabase account (for RAG database)
- Git (for repository cloning)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd pulsePal
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Environment Configuration

Create a `.env` file in the project root:

```env
# Required: LLM Configuration
GOOGLE_API_KEY=your_google_api_key_here
LLM_MODEL=gemini-2.0-flash-exp

# Required: Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_service_role_key

# Optional: BGE Model Path (uses default if not specified)
BGE_MODEL_PATH=/path/to/bge/model

# Optional: Session Configuration
MAX_SESSION_DURATION_HOURS=24
MAX_CONVERSATION_HISTORY=100
USE_HYBRID_SEARCH=true
```

### Step 5: Initialize RAG Database (Optional)

If you have RAG data to initialize:

```bash
python initialize_embeddings.py
```

### Step 6: Verify Installation

Test the CLI interface:

```bash
python run_pulsepal.py "Hello, can you help me with Pulseq?"
```

Test the web interface:

```bash
chainlit run chainlit_app.py
```

## Usage Guide

### Web Interface (Recommended)

The Chainlit web interface provides the best user experience with:
- Modern, responsive design
- Markdown rendering with syntax highlighting  
- Session persistence
- Real-time typing indicators

**Start the web interface:**

```bash
chainlit run chainlit_app.py
```

Then open your browser to `http://localhost:8000`

**Example interactions:**

```
🔬 Welcome to Pulsepal!

User: How do I create a spin echo sequence in MATLAB?
Pulsepal: [Searches documentation and provides MATLAB code example]

User: Can you convert that to Python?
Pulsepal: [Converts the sequence to Python using pypulseq]

User: Explain the physics behind T2 contrast
Pulsepal: [Delegates to MRI Expert for detailed physics explanation]
```

### Command Line Interface

For programmatic access or server environments:

**Single query:**
```bash
python run_pulsepal.py "How do I set up a gradient echo sequence?"
```

**Interactive mode:**
```bash
python run_pulsepal.py --interactive
```

**Help:**
```bash
python run_pulsepal.py --help
```

### Programming Integration

You can also integrate PulsePal directly into Python code:

```python
import asyncio
from pulsepal.main_agent import run_pulsepal, create_pulsepal_session

async def main():
    # Single query
    session_id, response = await run_pulsepal("How do I create a FLASH sequence?")
    print(response)
    
    # Session-based conversation
    session_id, deps = await create_pulsepal_session()
    session_id, response1 = await run_pulsepal("Create a spin echo in MATLAB", session_id)
    session_id, response2 = await run_pulsepal("Now convert it to Python", session_id)

asyncio.run(main())
```

## Core Components

### Pulsepal Main Agent

**File:** `pulsepal/main_agent.py`

The primary agent that handles:
- User query processing
- RAG searches for documentation
- Code generation in MATLAB/Python/Octave
- Agent delegation to MRI Expert
- Session and conversation management

**Key capabilities:**
- Maintains conversation context across interactions
- Detects user language preferences (defaults to MATLAB)
- Seamlessly integrates tool responses
- Provides working, tested code examples

### MRI Expert Agent

**File:** `pulsepal/mri_expert_agent.py`

Specialized sub-agent for physics explanations:
- Nuclear magnetic resonance principles
- K-space theory and trajectory analysis
- RF pulse design and optimization
- Gradient encoding and timing
- Scanner hardware limitations
- Safety considerations (SAR, dB/dt, PNS)

**Usage pattern:**
The MRI Expert is automatically consulted when users ask physics questions. The delegation is transparent - users always feel they're talking to "Pulsepal."

### RAG Service

**File:** `pulsepal/rag_service.py`

Provides comprehensive document and code search:

```python
from pulsepal.rag_service import get_rag_service

rag = get_rag_service()

# Search documentation
results = rag.perform_rag_query("spin echo sequence", match_count=5)

# Search code examples  
code_results = rag.search_code_examples("gradient echo MATLAB")

# Get available sources
sources = rag.get_available_sources()
```

### Session Management

**File:** `pulsepal/dependencies.py`

Handles conversation continuity:
- Automatic session creation and cleanup
- Conversation history management
- Language preference detection and persistence
- Code example storage and retrieval

### Tools System

**File:** `pulsepal/tools.py`

PydanticAI tools that the agents use:

- `perform_rag_query`: Search documentation
- `search_code_examples`: Find code implementations
- `get_available_sources`: Discover available documentation
- `delegate_to_mri_expert`: Consult physics expert

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | Yes | - | Google API key for Gemini |
| `LLM_MODEL` | No | `gemini-2.0-flash-exp` | Gemini model to use |
| `SUPABASE_URL` | Yes | - | Supabase project URL |
| `SUPABASE_KEY` | Yes | - | Supabase service role key |
| `BGE_MODEL_PATH` | No | Auto-detected | Path to BGE embedding model |
| `MAX_SESSION_DURATION_HOURS` | No | 24 | Session timeout |
| `USE_HYBRID_SEARCH` | No | true | Enable hybrid search |

### Settings Configuration

**File:** `pulsepal/settings.py`

The settings system uses Pydantic Settings for type-safe configuration:

```python
from pulsepal.settings import get_settings

settings = get_settings()
print(f"Using model: {settings.llm_model}")
print(f"Max session duration: {settings.max_session_duration_hours} hours")
```

### Chainlit Configuration

**File:** `chainlit.md`

Customize the web interface welcome message and branding by editing this file.

## API Reference

### Main Agent Functions

#### `run_pulsepal(query: str, session_id: str = None) -> tuple[str, str]`

Run Pulsepal agent with a query.

**Parameters:**
- `query`: User question about Pulseq programming
- `session_id`: Optional session ID for conversation continuity

**Returns:**
- `tuple`: (session_id, agent_response)

**Example:**
```python
session_id, response = await run_pulsepal("How do I create a FLASH sequence?")
```

#### `create_pulsepal_session(session_id: str = None) -> tuple[str, PulsePalDependencies]`

Create a new Pulsepal session with initialized dependencies.

**Parameters:**
- `session_id`: Optional session ID (generates UUID if not provided)

**Returns:**
- `tuple`: (session_id, initialized_dependencies)

#### `ask_pulsepal(query: str) -> str`

Simple interface without session management.

**Parameters:**
- `query`: User question

**Returns:**
- `str`: Agent response

### MRI Expert Functions

#### `consult_mri_expert(question: str, context: str = None) -> str`

Consult MRI Expert for physics explanations.

**Parameters:**
- `question`: Physics question to explain
- `context`: Additional context from conversation

**Returns:**
- `str`: Expert physics explanation

#### `explain_mri_concept(concept: str, detail_level: str = "intermediate") -> str`

Get explanation of specific MRI concept.

**Parameters:**
- `concept`: MRI concept (e.g., "T1 relaxation", "k-space")
- `detail_level`: "basic", "intermediate", or "advanced"

**Returns:**
- `str`: Concept explanation at appropriate level

### RAG Service API

#### `perform_rag_query(query: str, source: str = None, match_count: int = 5) -> str`

Search documentation database.

**Parameters:**
- `query`: Search query
- `source`: Optional source filter
- `match_count`: Number of results (1-20)

**Returns:**
- `str`: Formatted search results

#### `search_code_examples(query: str, source_id: str = None, match_count: int = 5) -> str`

Search code examples database.

**Parameters:**
- `query`: Code search query
- `source_id`: Optional source filter
- `match_count`: Number of results

**Returns:**
- `str`: Formatted code examples

### Session Management API

#### `SessionManager.create_session(session_id: str) -> ConversationContext`

Create new session with conversation context.

#### `SessionManager.get_session(session_id: str) -> ConversationContext`

Get existing session or create new one.

#### `ConversationContext.add_conversation(role: str, content: str)`

Add conversation entry with automatic history management.

#### `ConversationContext.detect_language_preference(content: str) -> str`

Detect language preference from user content.

## Testing

### Running Tests

PulsePal includes comprehensive test suites:

```bash
# Run all tests
pytest

# Run specific test categories
pytest pulsepal/tests/test_basic_functionality.py
pytest pulsepal/tests/test_agent_delegation.py
pytest pulsepal/tests/test_session_management.py
pytest pulsepal/tests/test_chainlit_integration.py

# Run with coverage
pytest --cov=pulsepal

# Run Chainlit-specific tests
python run_chainlit_tests.py
```

### Test Structure

- **Basic Functionality**: Tests core agent responses using TestModel
- **Agent Delegation**: Verifies MRI Expert integration
- **Session Management**: Tests conversation continuity
- **Chainlit Integration**: Tests web interface handlers
- **RAG Integration**: Tests documentation search functionality

### Using TestModel for Development

For development and testing, use PydanticAI's TestModel:

```python
from pydantic_ai.models.test import TestModel
from pulsepal.main_agent import pulsepal_agent

# Override with TestModel for testing
test_model = TestModel()
test_agent = pulsepal_agent.override(model=test_model)

# Test with controlled responses
result = await test_agent.run("test query", deps=test_deps)
```

## Deployment

### Local Development

```bash
# Start web interface for development
chainlit run chainlit_app.py --debug

# Or use CLI interface
python run_pulsepal.py --interactive
```

### Production Deployment

#### Docker Deployment (Recommended)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Expose Chainlit port
EXPOSE 8000

# Run Chainlit web interface
CMD ["chainlit", "run", "chainlit_app.py", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t pulsepal .
docker run -p 8000:8000 --env-file .env pulsepal
```

#### Cloud Deployment

For cloud platforms (AWS, GCP, Azure):

1. **Environment Variables**: Set all required environment variables
2. **Dependencies**: Ensure BGE model is accessible
3. **Database**: Configure Supabase connection
4. **Health Checks**: Implement health check endpoints
5. **Scaling**: Consider session persistence for multiple instances

#### Environment-Specific Configuration

**Development:**
```env
CHAINLIT_DEBUG=true
LOG_LEVEL=DEBUG
```

**Production:**
```env
CHAINLIT_DEBUG=false
LOG_LEVEL=INFO
USE_HYBRID_SEARCH=true
```

### Performance Considerations

- **Memory**: BGE model requires ~2GB RAM
- **CPU**: Embedding generation is CPU-intensive  
- **Network**: Supabase connection should be reliable
- **Caching**: RAG service includes built-in performance monitoring

## Development Guide

### Project Structure

```
pulsePal/
├── pulsepal/                    # Core package
│   ├── main_agent.py           # Main Pulsepal agent
│   ├── mri_expert_agent.py     # Physics expert agent
│   ├── tools.py                # Agent tools (RAG, delegation)
│   ├── dependencies.py         # Session management
│   ├── rag_service.py          # RAG functionality
│   ├── settings.py             # Configuration
│   ├── providers.py            # LLM provider setup
│   ├── supabase_client.py      # Database client
│   ├── embeddings.py           # BGE embedding service
│   └── tests/                  # Test suite
├── chainlit_app.py             # Web interface
├── run_pulsepal.py             # CLI interface
├── requirements.txt            # Dependencies
├── .env                        # Environment variables
└── examples/                   # Reference implementations
```

### Adding New Features

#### Adding New Tools

1. Define tool in `pulsepal/tools.py`:

```python
@get_agent().tool
async def new_tool(
    ctx: RunContext[PulsePalDependencies],
    parameter: str
) -> str:
    """Tool description."""
    # Implementation
    return result
```

2. Add validation model if needed:

```python
class NewToolParams(BaseModel):
    parameter: str = Field(..., description="Parameter description")
```

3. Update agent system prompt to mention the new tool

#### Adding New Agents

1. Create agent file (e.g., `pulsepal/new_agent.py`)
2. Define system prompt and capabilities
3. Create dependency class
4. Add delegation tool to main agent
5. Update tests

#### Extending RAG Capabilities

1. Add new search methods to `rag_service.py`
2. Implement corresponding tools
3. Update Supabase schema if needed
4. Add performance monitoring

### Code Style and Standards

- **Type Hints**: Use comprehensive type annotations
- **Async/Await**: Consistent async patterns throughout
- **Error Handling**: Graceful degradation with informative messages
- **Documentation**: Comprehensive docstrings and comments
- **Testing**: TestModel validation for all agent features

### Contributing Guidelines

1. **Branch Strategy**: Create feature branches from main
2. **Testing**: All new features must include tests
3. **Documentation**: Update documentation for new features
4. **Code Review**: Submit pull requests for review
5. **Environment Variables**: Never commit secrets to version control

### Troubleshooting

#### Common Issues

**"Failed to load settings" Error:**
- Check `.env` file exists and contains required variables
- Verify API keys are valid and not expired

**RAG Search Failures:**
- Verify Supabase connection and credentials
- Check BGE model path is correct
- Ensure network connectivity

**Agent Not Responding:**
- Check Google API key and quota
- Verify model name is correct
- Look for rate limiting issues

**Session Issues:**
- Clear browser cache for web interface
- Check session timeout settings
- Verify conversation context is being maintained

#### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or set environment variable:
```bash
export LOG_LEVEL=DEBUG
```

### Performance Optimization

- **RAG Caching**: Results are cached automatically
- **Embedding Reuse**: BGE model loads once per session
- **Session Cleanup**: Automatic cleanup of expired sessions
- **Memory Management**: Conversation history is trimmed automatically

---

## Support and Resources

- **Issues**: Report bugs via GitHub issues
- **Discussions**: Use GitHub discussions for questions
- **Documentation**: This documentation and inline code comments
- **Examples**: See `examples/` directory for reference implementations

**Version:** 1.0.0  
**Last Updated:** January 2025  
**License:** MIT License