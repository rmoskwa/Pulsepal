# Technical Constraints and Integration Requirements

## Existing Technology Stack

**Languages**: Python 3.10+
**Frameworks**: PydanticAI (>=0.0.11), Chainlit (>=1.1.0), uvicorn (>=0.25.0)
**Database**: Supabase (>=2.5.0) with pgvector for embeddings
**Infrastructure**: WSL2 environment, Google Cloud (Gemini API)
**External Dependencies**: Google Gemini 2.5 Flash, sentence-transformers (>=2.2.0)

## Integration Approach

**Database Integration Strategy**: No changes to Supabase schema or vector storage; maintain existing connection patterns through supabase_client.py

**API Integration Strategy**: Preserve all existing API endpoints; update internal routing only where module names change

**Frontend Integration Strategy**: Zero changes to Chainlit interface; maintain exact same message handling and streaming patterns

**Testing Integration Strategy**: Update test imports to match new module names; add pytest fixtures for v2 modules; implement test discovery for CI/CD

## Code Organization and Standards

**File Structure Approach**: Remove v2 suffixes after migration complete; consolidate debug modules into tests directory; maintain current package structure

**Naming Conventions**: Remove version suffixes (main_agent_v2 → main_agent); follow PEP 8 for all new code; maintain existing pattern names

**Coding Standards**: Follow existing CLAUDE.md guidelines; maintain current async/await patterns; preserve PydanticAI agent patterns

**Documentation Standards**: Update all docstrings to reflect current architecture; maintain Google-style docstrings; update README with current module names

## Deployment and Operations

**Build Process Integration**: Update requirements.txt if any dependencies change; ensure all imports resolve correctly; add pre-commit hooks for import validation

**Deployment Strategy**: Staged refactoring with module-by-module updates; test each change in CLI before Chainlit; maintain rollback capability

**Monitoring and Logging**: Implement log rotation for conversation logs; add metrics for session cleanup; maintain existing error logging patterns

**Configuration Management**: No changes to .env structure; maintain existing settings.py patterns; preserve API key management approach

## Risk Assessment and Mitigation

**Technical Risks**: Breaking changes during module renaming could crash production; incomplete test coverage might miss edge cases; circular dependencies might emerge during refactoring

**Integration Risks**: Chainlit interface might break with import changes (already broken on WSL2); Supabase connections could fail if client initialization changes; Session management changes might lose conversation history

**Deployment Risks**: No rollback strategy if refactoring fails; No CI/CD to catch issues before deployment; Manual testing only on WSL2 (no Chainlit testing possible)

**Mitigation Strategies**: Create comprehensive backup before starting; implement changes in small, testable increments; test each module rename in isolation; maintain parallel v2 files until migration verified; add integration tests before removing old modules
