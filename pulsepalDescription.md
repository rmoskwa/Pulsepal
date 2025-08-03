# Pulsepal Project Summary - v2.0 Intelligent Refactor

## Overview
**Pulsepal** is an intelligent AI assistant for MRI pulse sequence programming using the Pulseq framework. Built with PydanticAI, it combines comprehensive MRI physics knowledge with selective access to Pulseq documentation, providing fast, accurate responses without unnecessary searches.

## Architecture
- **Framework**: PydanticAI single-agent system with intelligent decision-making
- **LLM**: Google Gemini 2.5 Flash (enhanced reasoning capabilities)
- **Knowledge Base**: Supabase vector database with 25+ Pulseq documentation sources
- **Embeddings**: BGE-large-en-v1.5 for semantic search (local, no API costs)
- **Interfaces**: CLI (`run_pulsepal.py`) and Web UI (Chainlit - `chainlit_app.py`)

## Key Components

### 1. **Intelligent Agent**
- **Pulsepal Agent** (`pulsepal/main_agent.py`): Unified assistant with built-in MRI knowledge
  - Uses knowledge for general MRI physics and programming questions
  - Selectively searches only for Pulseq-specific implementations
  - Enhanced debugging with step-by-step reasoning

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

### 4. **Intelligent Tool System**
Unified tool in `pulsepal/tools.py`:
- `search_pulseq_knowledge`: Single intelligent tool that auto-routes queries
  - Detects query type (documentation/code/sources)
  - Only searches when Pulseq-specific details are needed
  - Graceful fallbacks: RAG → Web search → Helpful error

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
│   ├── main_agent.py       # Intelligent Pulsepal agent
│   ├── dependencies.py     # Session management
│   ├── rag_service.py      # RAG functionality
│   ├── supabase_client.py  # Database client
│   ├── tools.py           # Unified intelligent tool
│   └── settings.py        # Configuration
├── test_intelligence.py    # Intelligence validation suite
├── performance_metrics.py  # Performance tracking
├── validate_intelligence.py # Quick demo script
├── TEST_RESULTS.md        # Validation results
├── .env                    # Environment variables
└── requirements.txt        # Dependencies
```

## Key Features
1. **Intelligent decision-making** - knows when to search vs use knowledge
2. **Multi-language code generation** for MRI sequences
3. **Selective RAG search** only for Pulseq-specific implementations
4. **Built-in MRI physics knowledge** for instant explanations
5. **Enhanced debugging** with Gemini 2.5 Flash reasoning
6. **Session continuity** with conversation memory
7. **Direct PydanticAI integration** (no HTTP API layer)

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
- **Intelligent operation**: Agent decides when to search based on query content
- **Performance optimized**: 2-3 second faster responses for 90% of queries
- **Local embeddings**: Uses BGE model locally, no embedding API costs
- **Default language**: MATLAB unless user specifies otherwise

## v2.0 Improvements
- **Single agent architecture**: Removed redundant MRI Expert agent
- **80% fewer searches**: Only searches for Pulseq-specific details
- **50% less code**: Simplified from complex multi-agent to intelligent single agent
- **Enhanced debugging**: Better reasoning with Gemini 2.5 Flash
- **Faster responses**: Meets all performance targets (2-4s general, 3-6s specific)
