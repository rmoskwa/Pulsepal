# PulsePal Project Overview

## What is PulsePal?

PulsePal is an intelligent MRI sequence programming assistant that combines deep MRI physics knowledge with the Pulseq framework expertise. Built with Google Gemini 2.5 Flash and PydanticAI, it provides researchers and engineers with instant, accurate guidance for MRI sequence development.

## Key Features

- **Intelligence-First Design**: 90% of queries answered using built-in knowledge, selective RAG search only when needed
- **Multi-Language Support**: Generates code in MATLAB (default) and Python (pypulseq)
- **Session Management**: Maintains conversation context and language preferences
- **Dual Interface**: Both CLI and web UI (Chainlit) interfaces
- **Source-Aware RAG**: Intelligent routing to API docs, examples, or tutorials based on query intent

## Architecture Highlights

### Single Agent Design
PulsePal uses a monolithic intelligent agent architecture without sub-agents or delegation patterns. This design choice optimizes for:
- Faster response times (90% improvement over traditional RAG)
- Simpler maintenance and debugging
- Better context preservation

### Technology Stack
- **LLM**: Google Gemini 2.5 Flash
- **Framework**: PydanticAI
- **Vector Database**: Supabase with pgvector
- **Embeddings**: Google Embeddings API
- **Web UI**: Chainlit
- **CLI**: Native Python with argparse

## Quick Start

### Prerequisites
- Python 3.11+
- Google Cloud API key
- Supabase project credentials

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/pulsepal.git
cd pulsepal

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Basic Usage

#### CLI Mode
```bash
# Interactive mode
python run_pulsepal.py

# Single query
python run_pulsepal.py "How do I create a gradient echo sequence?"
```

#### Web UI
```bash
chainlit run chainlit_app.py
```

#### Python API
```python
from pulsepal.main_agent import run_pulsepal

session_id, response = await run_pulsepal("What is T1 relaxation?")
```

## Documentation Structure

- **[API_REFERENCE.md](./API_REFERENCE.md)** - Complete API documentation
- **[ARCHITECTURE_DIAGRAMS.md](./ARCHITECTURE_DIAGRAMS.md)** - Visual system architecture
- **[QUICK_REFERENCE.md](./QUICK_REFERENCE.md)** - Common commands and usage patterns
- **[SESSION_MANAGEMENT.md](./SESSION_MANAGEMENT.md)** - Session handling details

## Project Structure

```
pulsepal/
├── pulsepal/              # Core package
│   ├── main_agent.py     # Main PulsePal agent
│   ├── rag_service.py    # RAG service implementation
│   ├── dependencies.py   # Session management
│   ├── tools.py          # Agent tools
│   └── settings.py       # Configuration
├── chainlit_app.py       # Web UI interface
├── run_pulsepal.py       # CLI interface
├── docs/                 # Documentation
├── tests/                # Test suite
└── requirements.txt      # Dependencies
```

## Development Principles

### Intelligence-First Approach
- Trust Gemini's built-in knowledge for general MRI physics
- Use RAG search selectively for Pulseq-specific functions
- Function detection provides hints, not restrictions

### Fail-Fast Policy
- No graceful fallbacks that mask errors
- Clear error messages for debugging
- Explicit validation when needed

### Session Continuity
- Preserve conversation context across interactions
- Track language preferences automatically
- Maintain code examples within sessions

## Performance Characteristics

- **Response Time**: 1-3 seconds for built-in knowledge, 2-5 seconds with RAG
- **Accuracy**: 95%+ for common Pulseq patterns
- **Session Duration**: 24 hours (configurable)
- **Concurrent Users**: Unlimited (API rate limits apply)

## Testing

```bash
# Run test suite
pytest

# With coverage
pytest --cov=pulsepal

# Specific tests
pytest tests/test_main_agent.py
```

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/pulsepal/issues)
- **Email**: rmoskwa@wisc.edu
- **Documentation**: This directory

## License

MIT License - See LICENSE file for details
