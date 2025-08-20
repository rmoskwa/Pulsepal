# PulsePal Brownfield Architecture Document - Refactoring Focus

## Introduction

This document captures the CURRENT STATE of the PulsePal codebase as of January 2025, focusing on areas requiring refactoring. PulsePal has evolved through multiple iterations, resulting in redundant code, broken tests, and architectural inconsistencies that need addressing.

### Document Scope

This documentation is specifically focused on identifying refactoring opportunities including:
- Redundant and unused code modules
- Broken test infrastructure
- Version migration inconsistencies (v1 → v2)
- Architectural debt and workarounds
- Module coupling and dependency issues

### Change Log

| Date       | Version | Description                           | Author   |
| ---------- | ------- | ------------------------------------- | -------- |
| 2025-01-18 | 1.0     | Initial refactoring-focused analysis | Winston  |

## Quick Reference - Key Files and Entry Points

### Critical Active Files (Currently in Use)
- **Main CLI Entry**: `run_pulsepal.py` - CLI interface using v2 components
- **Web UI Entry**: `chainlit_app_v2.py` - Chainlit web interface
- **Core Agent**: `pulsepal/main_agent_v2.py` - Single intelligent agent
- **RAG Service**: `pulsepal/rag_service_v2.py` - Modern RAG with hybrid search
- **Tools Interface**: `pulsepal/tools_v2.py` - Unified tool interface
- **Configuration**: `pulsepal/settings.py` - Environment configuration
- **Dependencies**: `pulsepal/dependencies.py` - Session management

### Orphaned/Broken References (Need Removal)
- **Non-existent**: `pulsepal/main_agent.py` - Referenced by tests but doesn't exist
- **Non-existent**: `pulsepal/rag_service.py` - Referenced by tests but doesn't exist
- **Non-existent**: `pulsepal/tools.py` - Referenced by tests but doesn't exist

## High-Level Architecture

### Technical Summary

PulsePal is an MRI sequence programming assistant built with PydanticAI, designed to help researchers with Pulseq framework programming.

### Actual Tech Stack

| Category      | Technology           | Version      | Notes                                    |
| ------------- | -------------------- | ------------ | ---------------------------------------- |
| Runtime       | Python               | 3.10+        | WSL2 environment constraints             |
| LLM Framework | PydanticAI           | >=0.0.11     | Modern async agent framework             |
| LLM Model     | Google Gemini        | 2.5 Flash    | Primary intelligence engine              |
| Vector DB     | Supabase             | >=2.5.0      | pgvector for embeddings                  |
| Web UI        | Chainlit             | >=1.1.0      | Note: Broken on WSL2                     |
| Embeddings    | sentence-transformers| >=2.2.0      | For semantic routing                     |
| API Framework | uvicorn              | >=0.25.0     | ASGI server                              |

### Repository Structure Reality Check

- Type: Monorepo with test infrastructure issues
- Package Manager: pip/requirements.txt
- Notable Issues:
  - Tests reference non-existent v1 modules
  - Multiple versioning patterns (v2, _enhanced in cache)
  - Large conversation logs directory (500+ session files)
  - Documentation scattered across multiple locations

## Source Tree and Module Organization

### Project Structure (Actual with Issues)

```text
pulsePal/
├── pulsepal/                    # Main package directory
│   ├── main_agent_v2.py        # ✅ Active: Core agent
│   ├── rag_service_v2.py       # ✅ Active: RAG implementation
│   ├── tools_v2.py              # ✅ Active: Tool definitions
│   ├── dependencies.py          # ✅ Active: Session management
│   ├── settings.py              # ✅ Active: Configuration
│   ├── providers.py             # ✅ Active: LLM provider setup
│   ├── supabase_client.py      # ✅ Active: Database client
│   ├── semantic_router.py      # ✅ Active: Query routing
│   ├── startup.py               # ✅ Active: Service initialization
│   ├── conversation_logger.py  # ✅ Active: Logging utility
│   ├── auth.py                  # ✅ Active: API key authentication
│   ├── embeddings.py            # ✅ Active: Embedding generation
│   ├── gemini_patch.py         # ✅ Active: Gemini error handling
│   ├── markdown_fix_post_processor.py # ⚠️ Unknown usage
│   ├── timeout_utils.py        # ⚠️ Limited usage
│   ├── function_index.py       # 📊 Data: Function definitions
│   ├── source_profiles.py      # 📊 Data: Document profiles
│   ├── rag_formatters.py       # 🔧 Utility: Format RAG results
│   ├── code_validator.py       # 🧪 Used by tools_v2
│   ├── code_patterns.py        # 🧪 Used by debug_analyzer
│   ├── concept_mapper.py       # 🧪 Used by debug_analyzer
│   ├── debug_analyzer.py       # 🧪 Used by tests only
│   ├── syntax_validator.py     # ❓ Possibly unused
│   ├── recitation_monitor.py   # ❓ Possibly unused
│   ├── rag_optimization.py     # ❓ Possibly unused
│   ├── rag_performance.py      # ❓ Possibly unused
│   └── env_validator.py        # ❓ Possibly unused
├── tests/                       # ⚠️ BROKEN: Many tests import non-existent modules
│   ├── test_rag_v2.py         # ✅ Working: Uses v2 modules
│   ├── test_semantic_router.py # ✅ Working: Router tests
│   ├── test_code_patterns.py   # ✅ Working: Pattern tests
│   ├── test_debugging_flexibility.py # ✅ Working: Debug tests
│   ├── test_source_aware_rag.py     # ✅ Working: RAG tests
│   ├── test_function_discovery.py   # ❌ BROKEN: Imports non-existent tools
│   ├── test_function_verification.py # ❌ BROKEN: Imports non-existent tools, rag_service
│   ├── test_semantic_intent.py      # ❌ BROKEN: Imports non-existent main_agent
│   ├── test_rag_enhancement.py      # ❌ BROKEN: Imports non-existent rag_service
│   └── test_matlab_priority.py      # ❌ BROKEN: Imports non-existent rag_service
├── conversationLogs/            # 💾 500+ session files (needs cleanup strategy)
├── recrawlDatabase/             # 🔄 Database maintenance scripts
├── migrations/                  # 📁 Empty directory
├── matlab/                      # 📚 MATLAB integration examples
├── docs/                        # 📚 Documentation (mixed quality)
├── api-docs-deploy/            # 📚 API documentation deployment
├── site/                       # 📚 Generated MkDocs site
├── run_pulsepal.py             # ✅ Active: CLI entry point
├── chainlit_app_v2.py          # ✅ Active: Web UI entry point
├── generate_api_keys.py        # 🔧 Utility script
├── generate_matlab_docs.py     # 🔧 Utility script
├── requirements.txt            # ✅ Active: Dependencies
├── mkdocs.yml                  # 📚 Documentation config
└── CLAUDE.md                   # 📚 Development guidelines
```

## Technical Debt and Known Issues

### Critical Technical Debt

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

### Redundant/Unused Code Analysis

#### Definitely Unused (No imports found)
- `syntax_validator.py` - Syntax validation functionality
- `recitation_monitor.py` - Monitoring for recitation errors
- `rag_optimization.py` - RAG optimization utilities
- `rag_performance.py` - Performance monitoring
- `env_validator.py` - Environment validation

#### Used Only by Broken Tests
- `debug_analyzer.py` - Debug analysis (only by test_debugging_flexibility.py)
- Parts of `code_patterns.py` - Pattern analysis (limited usage)
- Parts of `concept_mapper.py` - Concept mapping (limited usage)

#### Unclear Usage Status
- `markdown_fix_post_processor.py` - Markdown processing
- `timeout_utils.py` - Timeout utilities

### Architectural Inconsistencies

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

## Migration Path from V1 to V2

### What Changed
- `main_agent.py` → `main_agent_v2.py`
- `rag_service.py` → `rag_service_v2.py`
- `tools.py` → `tools_v2.py`
- Multi-agent architecture → Single intelligent agent
- Complex routing → Simplified semantic routing

### What Broke
- All tests importing old module names
- Function discovery features referenced in tests
- Any documentation referencing old architecture

## Integration Points and External Dependencies

### External Services

| Service        | Purpose            | Integration Status | Issues                    |
| -------------- | ------------------ | ------------------ | ------------------------- |
| Google Gemini  | LLM Provider       | ✅ Working         | None                      |
| Supabase       | Vector DB          | ✅ Working         | Project ID: mnbvsrsivuuuwbtkmumt |
| Google Embeddings | Text embeddings | ✅ Working         | Separate API key needed   |
| Chainlit       | Web UI             | ⚠️ Partial         | Broken on WSL2            |

### Internal Integration Points

- **CLI Interface**: `run_pulsepal.py` → `main_agent_v2.py`
- **Web Interface**: `chainlit_app_v2.py` → `main_agent_v2.py`
- **Session Management**: Via `dependencies.py` and `SessionManager`
- **RAG Pipeline**: `main_agent_v2.py` → `tools_v2.py` → `rag_service_v2.py`

## Development and Testing Reality

### Current Testing Status

```bash
# Working tests (using v2 modules)
pytest tests/test_rag_v2.py              ✅
pytest tests/test_semantic_router.py      ✅
pytest tests/test_code_patterns.py        ✅
pytest tests/test_debugging_flexibility.py ✅

# Broken tests (importing non-existent modules)
pytest tests/test_function_discovery.py   ❌ ModuleNotFoundError: pulsepal.tools
pytest tests/test_function_verification.py ❌ ModuleNotFoundError: pulsepal.tools
pytest tests/test_semantic_intent.py      ❌ ModuleNotFoundError: pulsepal.main_agent
pytest tests/test_rag_enhancement.py      ❌ ModuleNotFoundError: pulsepal.rag_service
pytest tests/test_matlab_priority.py      ❌ ModuleNotFoundError: pulsepal.rag_service
```

### Local Development Setup Issues

1. Must use specific Python version (3.10+)
2. WSL2 users cannot test Chainlit interface
3. No clear documentation on which tests should work
4. Missing migration guide from v1 to v2

## Refactoring Recommendations

### Immediate Actions Needed

1. **Fix or Remove Broken Tests**:
   - Update imports to use v2 modules
   - Or remove tests for deprecated functionality
   - Priority: High - Tests provide no value when broken

2. **Clean Up Unused Modules**:
   - Remove definitely unused modules (see list above)
   - Archive or document modules of unclear status
   - Priority: Medium - Reduces maintenance burden

3. **Standardize Naming**:
   - Remove v2 suffix (these are now the primary versions)
   - Rename files to be more descriptive
   - Priority: Low - Cosmetic but improves clarity

4. **Session Log Management**:
   - Implement rotation/archival strategy
   - Add cleanup utility or automatic cleanup
   - Priority: Medium - Prevents disk space issues

5. **Documentation Update**:
   - Update all references to old module names
   - Create migration guide for v1 → v2
   - Document which tests are expected to work
   - Priority: High - Critical for maintenance

### Module-Specific Refactoring Needs

| Module | Current State | Recommendation | Priority |
| ------ | ------------ | -------------- | -------- |
| `main_agent_v2.py` | Active, well-structured | Rename to `main_agent.py` | Low |
| `rag_service_v2.py` | Active, complex | Rename, consider splitting | Medium |
| `tools_v2.py` | Active | Rename to `tools.py` | Low |
| `debug_analyzer.py` | Only used by tests | Move to tests/ or remove | Medium |
| `code_patterns.py` | Limited use | Evaluate necessity | Low |
| `concept_mapper.py` | Limited use | Evaluate necessity | Low |
| `syntax_validator.py` | Unused | Remove | High |
| `recitation_monitor.py` | Unused | Remove | High |
| `rag_optimization.py` | Unused | Remove | High |
| `rag_performance.py` | Unused | Remove | High |
| `env_validator.py` | Unused | Remove | High |

## Performance and Optimization Opportunities

### Current Performance Characteristics
- Semantic router loads 80MB model (cached after first load)
- RAG service uses hybrid search (vector + keyword)
- Session management maintains conversation history
- No apparent caching of RAG results

### Optimization Opportunities
1. Implement RAG result caching
2. Optimize conversation history storage
3. Consider lazy loading for heavy modules
4. Add performance monitoring to identify bottlenecks

## Security and Maintenance Concerns

### Security Issues
- API keys stored in `alpha_keys.json` (should be in .env)
- Conversation logs may contain sensitive data
- No apparent rate limiting beyond auth module

### Maintenance Issues
- No automated testing in CI/CD
- No code coverage metrics
- Inconsistent error handling patterns
- Missing logging in some modules

## Appendix - Useful Commands and Scripts

### Working Commands
```bash
# Run CLI interface
python run_pulsepal.py "Your question about Pulseq"
python run_pulsepal.py --interactive

# Run working tests
pytest tests/test_rag_v2.py
pytest tests/test_semantic_router.py

# Generate API keys
python generate_api_keys.py

# Generate MATLAB docs
python generate_matlab_docs.py
```

### Debugging Commands
```bash
# Check which modules are actually imported
grep -r "from pulsepal\." --include="*.py" | grep -v test | grep -v __pycache__

# Find unused files
for f in pulsepal/*.py; do 
  name=$(basename $f .py)
  count=$(grep -r "from pulsepal\.$name\|from \.$name" --include="*.py" | wc -l)
  echo "$name: $count imports"
done

# Check test status
python -m pytest tests/ --co -q
```

### Migration Path
To migrate this codebase to a cleaner state:
1. Fix all broken test imports
2. Remove confirmed unused modules
3. Rename v2 files to remove version suffix
4. Update all imports and documentation
5. Implement session log rotation
6. Add CI/CD with test automation

---

## Summary for Refactoring PRD

This brownfield analysis reveals a codebase in transition from v1 to v2 architecture, with incomplete migration leaving broken tests and unused modules. The core functionality works but is obscured by technical debt. Key refactoring priorities:

1. **Test Infrastructure**: 5+ test files broken due to outdated imports
2. **Dead Code**: 5-10 modules confirmed unused or barely used
3. **Naming Consistency**: Remove v2 suffixes now that v1 is gone
4. **Session Management**: 500+ log files need cleanup strategy
5. **Documentation**: Update to reflect current architecture

The refactoring effort should focus on cleaning up the incomplete v1→v2 migration, removing dead code, and establishing clear module boundaries for maintainability.