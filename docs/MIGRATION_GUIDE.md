# PulsePal v1 to v2 Migration Guide

## Overview

PulsePal has undergone a significant architectural refactoring to improve performance and maintainability. The system has transitioned from a multi-agent architecture to a single intelligent agent design, resulting in 90% faster response times through intelligent decision-making.

## Breaking Changes

### Module Renames

The following modules have been renamed to remove the v2 suffix:

| Old Import | New Import | Description |
|------------|------------|-------------|
| `pulsepal.main_agent_v2` | `pulsepal.main_agent` | Core PulsePal agent |
| `pulsepal.rag_service_v2` | `pulsepal.rag_service` | RAG search service |
| `pulsepal.tools_v2` | `pulsepal.tools` | Tool interfaces |

### Removed Modules

The following modules have been removed or integrated into the main agent:

| Removed Module | Reason | Alternative |
|----------------|--------|-------------|
| `pulsepal.syntax_validator` | Functionality integrated | Built into main agent validation |
| `pulsepal.recitation_monitor` | No longer needed | Session management handles context |
| `pulsepal.code_patterns` | Simplified architecture | Pattern detection in main agent |
| `pulsepal.concept_mapper` | Integrated into RAG | RAG service handles concept mapping |
| `pulsepal.debug_analyzer` | Moved to main agent | Debugging logic in main agent |
| `pulsepal.env_validator` | Replaced by pydantic-settings | Settings validation automatic |
| `pulsepal.markdown_fix_post_processor` | No longer needed | Direct markdown generation |
| `pulsepal.rag_optimization` | Integrated into RAG service | Optimization built into rag_service |
| `pulsepal.rag_performance` | Merged with RAG service | Performance monitoring in rag_service |
| `pulsepal.timeout_utils` | Simplified timeout handling | Standard asyncio timeouts |

## Architecture Changes

### From Multi-Agent to Single Agent

**v1 Architecture:**
- Multiple specialized agents for different tasks
- Complex agent delegation and routing
- Higher latency due to agent communication overhead

**v2 Architecture:**
- Single intelligent agent with comprehensive capabilities
- Built-in knowledge vs. selective RAG search decision-making
- 90% performance improvement through intelligent routing

### Session Management

**New in v2:**
- Automatic session creation and persistence
- Conversation context maintained across interactions
- Language preference detection (MATLAB vs Python)
- Configurable session duration (default 24 hours)

## Migration Steps

### 1. Update All Imports

Search and replace all old module imports:

```python
# Old
from pulsepal.main_agent_v2 import pulsepal_agent
from pulsepal.rag_service_v2 import ModernPulseqRAG
from pulsepal.tools_v2 import search_pulseq_knowledge

# New
from pulsepal.main_agent import pulsepal_agent
from pulsepal.rag_service import ModernPulseqRAG
from pulsepal.tools import search_pulseq_knowledge
```

### 2. Remove References to Deleted Modules

Remove any imports or usage of deleted modules:

```python
# Remove these imports
from pulsepal.syntax_validator import SyntaxValidator  # No longer exists
from pulsepal.recitation_monitor import RecitationMonitor  # No longer exists
from pulsepal.code_patterns import CodePatternAnalyzer  # No longer exists
```

### 3. Update Configuration

Environment variables remain the same, but configuration is now handled through `pulsepal.settings`:

```python
# Old way (if using direct env vars)
import os
api_key = os.getenv('GOOGLE_API_KEY')

# New way (recommended)
from pulsepal.settings import get_settings
settings = get_settings()
api_key = settings.google_api_key
```

### 4. Update Tool Usage

Tool interfaces have been simplified:

```python
# Old (if using multiple tools)
from pulsepal.tools_v2 import search_documentation, search_functions, search_examples

# New (unified interface)
from pulsepal.tools import search_pulseq_knowledge
# The search type is automatically determined or can be specified
result = await search_pulseq_knowledge(query, search_type="auto")
```

### 5. Test Your Integration

After migration, test your integration:

```bash
# Run basic test
python -c "from pulsepal.main_agent import pulsepal_agent; print('Import successful')"

# Test with a simple query
python run_pulsepal.py "What is a gradient echo sequence?"
```

## API Changes

### Agent Interface

The main agent interface remains largely the same:

```python
# Both versions work the same way
from pulsepal.main_agent import pulsepal_agent, run_pulsepal

async def get_response(query: str):
    session_id, response = await run_pulsepal(query)
    return response
```

### RAG Service

RAG service now includes built-in optimization:

```python
# v2 - Simplified interface
from pulsepal.rag_service import ModernPulseqRAG

rag = ModernPulseqRAG()
results = await rag.search_pulseq_knowledge(query, search_type="auto")
```

## Configuration Migration

### Environment Variables

No changes to environment variable names:
- `GOOGLE_API_KEY` - Still required for Gemini
- `SUPABASE_URL` - Still required for vector database
- `SUPABASE_KEY` - Still required for vector database
- `LLM_MODEL` - Still defaults to "gemini-2.5-flash"

### New Configuration Options

v2 adds new configuration options in `.env`:
```bash
# Session management (new in v2)
MAX_SESSION_DURATION_HOURS=24
MAX_CONVERSATION_HISTORY=100

# Search optimization (new in v2)
USE_HYBRID_SEARCH=true
```

## Troubleshooting

### Common Issues

1. **ImportError for v2 modules**
   - Solution: Update imports to remove v2 suffix

2. **Missing module errors for deleted modules**
   - Solution: Remove imports and refactor to use main agent

3. **Session not persisting**
   - Solution: Ensure session_id is passed between requests

4. **RAG search not working**
   - Solution: Check Supabase credentials in .env

### Verification Checklist

- [ ] All imports updated to new module names
- [ ] No references to deleted modules
- [ ] Environment variables configured
- [ ] Basic import test passes
- [ ] Simple query test works
- [ ] Session management functioning

## Rollback Instructions

If you need to rollback to v1:

1. Restore old module files from git history
2. Revert import changes in your code
3. Restore v1 dependencies in requirements.txt
4. Clear any v2 session data

```bash
# Example rollback
git checkout <v1-commit-hash> -- pulsepal/
pip install -r requirements.v1.txt  # If you saved v1 requirements
```

## Support

For migration assistance or issues:
- Contact: rmoskwa@wisc.edu
- GitHub Issues: https://github.com/rmoskwa/Pulsepal/issues

## Version History

- **v2.0.0** (Current) - Single agent architecture with intelligent routing
- **v1.0.0** - Initial multi-agent implementation