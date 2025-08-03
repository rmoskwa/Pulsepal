# Pulsepal Project Summary

## Overview
**Pulsepal** is a sophisticated AI-powered assistant for MRI pulse sequence programming using the Pulseq framework. It's built as a PydanticAI multi-agent system (NOT a web API) that helps researchers and programmers work with Pulseq v1.5.0 across multiple programming languages.

## Architecture
- **Framework**: PydanticAI-based multi-agent system
- **LLM**: Google Gemini (gemini-2.0-flash-lite)
- **Knowledge Base**: Supabase vector database with 25+ Pulseq documentation sources
- **Embeddings**: BGE-large-en-v1.5 for semantic search
- **Interfaces**: CLI (`run_pulsepal.py`) and Web UI (Chainlit - `chainlit_app.py`)

## Key Components

### 1. **Agent**
- **Pulsepal Agent** (`pulsepal/main_agent.py`): Main assistant for code generation, debugging, documentation search, and MRI physics explanations

### 2. **RAG System**
- **Database**: Supabase project "crawl4ai-mcp-local-embed"
- **Tables**: 
  - `sources`: Metadata about documentation sources
  - `crawled_pages`: Chunked documentation with embeddings
  - `code_examples`: Code snippets with AI summaries
  - `api_reference`: **Contains supported Pulseq languages (including C/C++)**
- **Search**: Hybrid search combining vector similarity and keyword matching

### 3. **Supported Languages**
Currently implemented:
- MATLAB (default)
- Python (pypulseq)
- Octave

**Important**: The Supabase `api_reference` table indicates C/C++ support exists in the knowledge base, but the current UI implementation doesn't fully support C/C++ syntax highlighting and detection yet.

### 4. **Tools**
RAG tools implemented in `pulsepal/tools.py`:
- `perform_rag_query`: Search documentation
- `search_code_examples`: Find code implementations
- `get_available_sources`: List documentation sources

## Current Status
- ✅ CLI interface working
- ✅ Chainlit web UI functional
- ✅ RAG integration operational
- ✅ Session management implemented
- ⚠️ C/C++ support needs UI enhancement (exists in knowledge base but not in UI)

## File Structure
```
pulsePal/
├── chainlit_app.py         # Web UI using Chainlit
├── run_pulsepal.py         # CLI interface
├── pulsepal/
│   ├── main_agent.py       # Pulsepal agent
│   ├── dependencies.py     # Session management
│   ├── rag_service.py      # RAG functionality
│   ├── supabase_client.py  # Database client
│   ├── tools.py           # Agent tools
│   └── settings.py        # Configuration
├── .env                    # Environment variables
└── requirements.txt        # Dependencies
```

## Key Features
1. **Multi-language code generation** for MRI sequences
2. **Advanced RAG search** across comprehensive Pulseq documentation
3. **Comprehensive MRI physics knowledge** for theoretical explanations
4. **Session continuity** with conversation memory
5. **Direct PydanticAI integration** (no HTTP API layer)

## Environment Variables
```env
GOOGLE_API_KEY=<api_key>
SUPABASE_URL=https://mnbvsrsivuuuwbtkmumt.supabase.co
SUPABASE_KEY=<service_role_key>
BGE_MODEL_PATH=/path/to/embeddings/model
```

## Knowledge Base Sources
The Supabase database contains documentation from 25+ sources including:
- Official Pulseq repositories and documentation
- PyPulseq (Python implementation)
- HarmonizedMRI organization repositories
- Specialized sequence implementations (MOLLI, 3DEPI, SMS-EPI, etc.)

## Technical Notes
- **No FastAPI**: This is a pure PydanticAI system, not a web API
- **Direct integration**: Chainlit directly imports and calls PydanticAI agents
- **MCP Server**: User has Supabase MCP configured in Claude Desktop but it's not directly accessible to assistants
- **Default language**: MATLAB unless user specifies otherwise
