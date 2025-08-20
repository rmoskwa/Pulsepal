# PulsePal Quick Reference Guide

## Common CLI Commands

### Basic Usage
```bash
# Start interactive mode
python run_pulsepal.py

# Single query
python run_pulsepal.py "How do I create a spin echo sequence?"

# Continue session
python run_pulsepal.py --session-id abc123

# Get help
python run_pulsepal.py --help
```

### Session Management
```bash
# Clean up old sessions
python -c "from pulsepal.dependencies import SessionManager; SessionManager().cleanup_old_sessions()"

# List sessions (if CLI extensions installed)
python -m pulsepal.cli sessions list

# Export session history
python -m pulsepal.cli sessions export <session_id> --output history.json
```

### Web Interface
```bash
# Start Chainlit UI (not on WSL2)
chainlit run chainlit_app.py

# With custom port
chainlit run chainlit_app.py --port 8080
```

## Frequent API Calls

### Basic Query
```python
from pulsepal.main_agent import run_pulsepal

# Simple query
session_id, response = await run_pulsepal("What is T1 relaxation?")

# Follow-up question
_, response = await run_pulsepal("How does it affect contrast?", session_id)
```

### RAG Search
```python
from pulsepal.rag_service import ModernPulseqRAG

rag = ModernPulseqRAG()

# Auto search (intelligent routing)
results = await rag.search_pulseq_knowledge("gradient echo")

# Specific search types
docs = await rag.search_pulseq_knowledge(query, search_type="documentation")
funcs = await rag.search_pulseq_knowledge(query, search_type="functions")
examples = await rag.search_pulseq_knowledge(query, search_type="examples")
```

### Session Management
```python
from pulsepal.dependencies import SessionManager

manager = SessionManager()

# Create session
session_id = manager.create_session()

# Get session
context = manager.get_session(session_id)

# Clean old sessions
manager.cleanup_old_sessions()
```

## Debugging Tips

### Check Environment
```bash
# Verify environment variables
python -c "from pulsepal.settings import get_settings; print(get_settings())"

# Test imports
python -c "import pulsepal.main_agent; print('✓ Imports work')"
```

### Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run your code - will show detailed logs
```

### Common Issues

| Issue | Solution |
|-------|----------|
| Import error | `pip install -r requirements.txt` |
| API key missing | Check `.env` file has `GOOGLE_API_KEY` |
| Supabase error | Verify `SUPABASE_URL` and `SUPABASE_KEY` |
| Session not found | Session expired, create new one |
| Rate limit | Add delay between requests |
| Memory error | Reduce `MAX_CONVERSATION_HISTORY` |

### Test Connection
```python
# Test Gemini connection
from pulsepal.settings import get_llm_model
model = get_llm_model()
print("✓ Gemini connected")

# Test Supabase
from pulsepal.rag_service import get_supabase_client
client = get_supabase_client()
print("✓ Supabase connected")
```

## Performance Optimization Hints

### Speed Tips
1. **Use session IDs** - Maintains context, avoids re-initialization
2. **Specific search types** - Faster than auto-detection
3. **Limit conversation history** - Set `MAX_CONVERSATION_HISTORY=50`
4. **Enable caching** - RAG results are cached automatically

### Memory Management
```python
# Clear old sessions regularly
from pulsepal.dependencies import SessionManager
SessionManager().cleanup_old_sessions(max_age_hours=12)

# Limit session history
settings.max_conversation_history = 50
```

### Batch Processing
```python
# Process multiple queries efficiently
async def batch_process(queries):
    session_id = None
    results = []
    
    for query in queries:
        session_id, response = await run_pulsepal(query, session_id)
        results.append(response)
    
    return results
```

## Troubleshooting Checklist

### Initial Setup
- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file created from `.env.example`
- [ ] API keys configured

### Not Working?
1. **Check logs**: Look in `conversationLogs/` directory
2. **Verify settings**: `python -c "from pulsepal.settings import get_settings; get_settings()"`
3. **Test modules**: `python -c "import pulsepal.main_agent"`
4. **Check services**: Ensure Gemini and Supabase are accessible
5. **Review session**: Sessions expire after 24 hours by default

### Error Messages

| Error | Meaning | Fix |
|-------|---------|-----|
| `ConfigurationError` | Missing env vars | Check `.env` file |
| `SessionNotFoundError` | Session expired | Create new session |
| `RAGServiceError` | Search failed | Check Supabase connection |
| `RateLimitError` | Too many requests | Add delays or upgrade API |
| `TimeoutError` | Request too slow | Retry or simplify query |

## Contact/Support Information

### Getting Help
- **Email**: rmoskwa@wisc.edu
- **GitHub Issues**: https://github.com/rmoskwa/Pulsepal/issues
- **Documentation**: See `/docs` directory

### Reporting Bugs
Include:
1. Error message
2. Python version (`python --version`)
3. PulsePal version
4. Steps to reproduce
5. Session ID if applicable

### Feature Requests
Submit via GitHub Issues with:
- Use case description
- Expected behavior
- Current workaround (if any)

## Common Pulseq Questions

### Quick Answers

**Q: How do I create a basic gradient?**
```matlab
gx = mr.makeTrapezoid('x', lims, 'Amplitude', 1000, 'FlatTime', 3e-3);
```

**Q: What's the difference between `mr.*` and `seq.*`?**
- `mr.*` - Sequence building functions (makeTrapezoid, makeBlockPulse)
- `seq.*` - Sequence object methods (addBlock, write)

**Q: How do I check my sequence?**
```matlab
[ok, error_report] = seq.checkTiming();
seq.plot();  % Visualize
```

**Q: Common parameter mistakes?**
- Using `mr.write()` instead of `seq.write()`
- Missing required parameters in `make*` functions
- Wrong units (Hz vs rad/s, seconds vs milliseconds)

## Useful Code Snippets

### Initialize Sequence
```matlab
% MATLAB
seq = mr.Sequence();
lims = mr.opts('MaxGrad', 32, 'GradUnit', 'mT/m', ...
               'MaxSlew', 130, 'SlewUnit', 'T/m/s');
```

```python
# Python (pypulseq)
import pypulseq as pp
seq = pp.Sequence()
lims = pp.Opts(max_grad=32, grad_unit='mT/m',
               max_slew=130, slew_unit='T/m/s')
```

### Create RF Pulse
```matlab
% MATLAB
rf = mr.makeBlockPulse(flip_angle*pi/180, lims, ...
                       'Duration', 2e-3, 'system', lims);
```

### Add Gradient
```matlab
% MATLAB
gx = mr.makeTrapezoid('x', lims, 'Area', area);
seq.addBlock(gx);
```

### Write Sequence
```matlab
% MATLAB
seq.write('my_sequence.seq');
```

## Environment Variables Reference

```bash
# Required
GOOGLE_API_KEY=            # Google Gemini API key
SUPABASE_URL=              # Supabase project URL
SUPABASE_KEY=              # Supabase anon key

# Optional
LLM_MODEL=gemini-2.5-flash # Model selection
MAX_SESSION_DURATION_HOURS=24
MAX_CONVERSATION_HISTORY=100
USE_HYBRID_SEARCH=true
SESSION_STORAGE_PATH=./sessions
GOOGLE_API_KEY_EMBEDDING=   # For embeddings

# Logging
LOG_LEVEL=INFO             # DEBUG, INFO, WARNING, ERROR
LOG_FILE=pulsepal.log      # Log file path
```

## Quick Testing

```bash
# Test everything works
python -c "
from pulsepal.main_agent import run_pulsepal
import asyncio
result = asyncio.run(run_pulsepal('test'))
print('✓ All systems operational')
"
```