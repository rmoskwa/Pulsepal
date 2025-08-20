# Technical Debt and Known Issues

## Critical Technical Debt

1. **Version Migration Incomplete**:
   - Files renamed to `_v2.py` but old imports remain in tests
   - No `main_agent.py`, `rag_service.py`, or `tools.py` exist anymore
   - Tests are broken due to missing module references
   - Cache shows `_enhanced` versions suggesting multiple refactoring attempts

2. **Test Infrastructure Broken**:
   - 5+ test files import non-existent modules
   - Tests haven't been updated to use v2 modules
   - No CI/CD to catch these issues early

3. **Module Organization Issues**:
   - Unclear separation between active and inactive modules
   - Several modules of unknown usage status
   - Debug/analysis modules only used by broken tests

4. **Session Management Overhead**:
   - 500+ conversation log files accumulating
   - No cleanup or archival strategy
   - Each session creates 2-3 files (.jsonl, .txt, _searches.jsonl)

5. **WSL2 Compatibility**:
   - Chainlit doesn't work on WSL2 (documented in CLAUDE.md)
   - Testing limited to CLI interface on WSL2

## Redundant/Unused Code Analysis

### Definitely Unused (No imports found)
- `syntax_validator.py` - Syntax validation functionality
- `recitation_monitor.py` - Monitoring for recitation errors
- `rag_optimization.py` - RAG optimization utilities
- `rag_performance.py` - Performance monitoring
- `env_validator.py` - Environment validation

### Used Only by Broken Tests
- `debug_analyzer.py` - Debug analysis (only by test_debugging_flexibility.py)
- Parts of `code_patterns.py` - Pattern analysis (limited usage)
- Parts of `concept_mapper.py` - Concept mapping (limited usage)

### Unclear Usage Status
- `markdown_fix_post_processor.py` - Markdown processing
- `timeout_utils.py` - Timeout utilities

## Architectural Inconsistencies

1. **Naming Convention Issues**:
   - Mix of `_v2` suffix pattern (main_agent_v2.py)
   - No clear versioning strategy
   - Some modules have descriptive names, others are generic

2. **Import Pattern Inconsistencies**:
   - Some modules use relative imports (`.settings`)
   - Others use absolute imports (`pulsepal.settings`)
   - Circular dependency potential in tools registration

3. **Single vs Multi-Agent Confusion**:
   - CLAUDE.md emphasizes "Single Agent Design"
   - But run_pulsepal.py docstring mentions "multi-agent system"
   - Architecture has clearly evolved from multi to single agent
