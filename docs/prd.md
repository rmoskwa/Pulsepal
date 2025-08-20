# PulsePal Technical Debt Elimination Brownfield Enhancement PRD

## Intro Project Analysis and Context

### Existing Project Overview

#### Analysis Source
- Architecture document available at: `docs/brownfield-architecture-refactoring.md`

#### Current Project State
PulsePal is an MRI sequence programming assistant built with PydanticAI, using Google Gemini 2.5 Flash as the LLM engine and Supabase for vector storage. The system has evolved from a multi-agent to single-agent architecture, leaving incomplete migration artifacts, broken tests, and redundant code modules that need cleanup.

### Documentation Analysis

#### Available Documentation
- ✓ Tech Stack Documentation (from architecture doc)
- ✓ Source Tree/Architecture (detailed in architecture doc)
- ✓ Coding Standards (CLAUDE.md provides guidelines)
- ✓ API Documentation (api-docs-deploy directory)
- ✗ External API Documentation (partial)
- ✗ UX/UI Guidelines (not documented)
- ✓ Technical Debt Documentation (comprehensive in architecture doc)

### Enhancement Scope Definition

#### Enhancement Type
- ✓ Bug Fix and Stability Improvements (broken tests)
- ✓ Technology Stack Upgrade (completing v1→v2 migration)
- ✓ Performance/Scalability Improvements (session management)

#### Enhancement Description
Complete the unfinished v1→v2 migration, remove dead code, fix broken test infrastructure, standardize naming conventions, and implement proper session log management to reduce technical debt and improve maintainability.

#### Impact Assessment
- ✓ Significant Impact (substantial existing code changes required for test fixes, module removal, and renaming)

### Goals and Background Context

#### Goals
- Complete the v1→v2 migration cleanly
- Restore test suite to working condition
- Remove all unused and orphaned code modules
- Standardize module naming conventions
- Implement session log rotation strategy
- Update documentation to reflect current architecture

#### Background Context
This refactoring is critical because PulsePal has accumulated significant technical debt from an incomplete architectural transition. The codebase evolved from a multi-agent to single-agent design, but the migration was never fully completed. This has left broken tests that reference non-existent modules, redundant code that adds maintenance burden, and inconsistent naming that confuses developers. The refactoring will establish a clean, maintainable foundation for future development.

### Change Log

| Change | Date | Version | Description | Author |
|--------|------|---------|-------------|--------|
| Initial PRD | 2025-01-18 | 1.0 | Refactoring PRD based on architecture analysis | John (PM) |

## Requirements

### Functional Requirements

- **FR1**: The system shall update all test imports to reference the correct v2 modules (main_agent_v2, rag_service_v2, tools_v2)
- **FR2**: The system shall remove all confirmed unused modules (syntax_validator, recitation_monitor, rag_optimization, rag_performance, env_validator)
- **FR3**: The system shall rename v2 modules to remove version suffixes once v1 references are eliminated
- **FR4**: The system shall implement automatic session log rotation to prevent unlimited growth of conversationLogs directory
- **FR5**: The system shall update all documentation to reflect the current single-agent architecture
- **FR6**: The system shall ensure all tests in the test suite pass successfully
- **FR7**: The system shall maintain backward compatibility for existing API endpoints and CLI interfaces

### Non-Functional Requirements

- **NFR1**: All refactoring changes must maintain existing performance characteristics with response times under 2 seconds for typical queries
- **NFR2**: The refactored codebase must maintain 100% backward compatibility with existing Chainlit and CLI interfaces
- **NFR3**: Test execution time should not increase by more than 20% after refactoring
- **NFR4**: Session log storage should not exceed 1GB with automatic cleanup of logs older than 30 days
- **NFR5**: Code coverage should reach at least 80% after test fixes are complete

### Compatibility Requirements

- **CR1**: All existing API endpoints must continue to function without changes to request/response formats
- **CR2**: Database schema must remain unchanged to maintain compatibility with existing Supabase vector storage
- **CR3**: UI/UX must remain consistent - no changes to Chainlit interface or CLI command structure
- **CR4**: Integration with Google Gemini API and Supabase must maintain current authentication and configuration patterns

## Technical Constraints and Integration Requirements

### Existing Technology Stack

**Languages**: Python 3.10+
**Frameworks**: PydanticAI (>=0.0.11), Chainlit (>=1.1.0), uvicorn (>=0.25.0)
**Database**: Supabase (>=2.5.0) with pgvector for embeddings
**Infrastructure**: WSL2 environment, Google Cloud (Gemini API)
**External Dependencies**: Google Gemini 2.5 Flash, sentence-transformers (>=2.2.0)

### Integration Approach

**Database Integration Strategy**: No changes to Supabase schema or vector storage; maintain existing connection patterns through supabase_client.py

**API Integration Strategy**: Preserve all existing API endpoints; update internal routing only where module names change

**Frontend Integration Strategy**: Zero changes to Chainlit interface; maintain exact same message handling and streaming patterns

**Testing Integration Strategy**: Update test imports to match new module names; add pytest fixtures for v2 modules; implement test discovery for CI/CD

### Code Organization and Standards

**File Structure Approach**: Remove v2 suffixes after migration complete; consolidate debug modules into tests directory; maintain current package structure

**Naming Conventions**: Remove version suffixes (main_agent_v2 → main_agent); follow PEP 8 for all new code; maintain existing pattern names

**Coding Standards**: Follow existing CLAUDE.md guidelines; maintain current async/await patterns; preserve PydanticAI agent patterns

**Documentation Standards**: Update all docstrings to reflect current architecture; maintain Google-style docstrings; update README with current module names

### Deployment and Operations

**Build Process Integration**: Update requirements.txt if any dependencies change; ensure all imports resolve correctly; add pre-commit hooks for import validation

**Deployment Strategy**: Staged refactoring with module-by-module updates; test each change in CLI before Chainlit; maintain rollback capability

**Monitoring and Logging**: Implement log rotation for conversation logs; add metrics for session cleanup; maintain existing error logging patterns

**Configuration Management**: No changes to .env structure; maintain existing settings.py patterns; preserve API key management approach

### Risk Assessment and Mitigation

**Technical Risks**: Breaking changes during module renaming could crash production; incomplete test coverage might miss edge cases; circular dependencies might emerge during refactoring

**Integration Risks**: Chainlit interface might break with import changes (already broken on WSL2); Supabase connections could fail if client initialization changes; Session management changes might lose conversation history

**Deployment Risks**: No rollback strategy if refactoring fails; No CI/CD to catch issues before deployment; Manual testing only on WSL2 (no Chainlit testing possible)

**Mitigation Strategies**: Create comprehensive backup before starting; implement changes in small, testable increments; test each module rename in isolation; maintain parallel v2 files until migration verified; add integration tests before removing old modules

## Epic and Story Structure

### Epic Approach

**Epic Structure Decision**: Single comprehensive epic for the refactoring effort with rationale: This refactoring represents a cohesive technical debt reduction effort where all changes are interconnected. The v1→v2 migration, test fixes, and code cleanup must be coordinated to avoid breaking the system.

## Epic 1: PulsePal Technical Debt Elimination and v2 Migration Completion

**Epic Goal**: Complete the v1→v2 migration, eliminate technical debt, and establish a clean, maintainable codebase with working tests and proper session management

**Integration Requirements**: All changes must maintain backward compatibility with existing interfaces; each story must verify existing functionality remains intact

### Story 1.1: Fix Test Infrastructure and Update Imports

As a developer,
I want to update all test files to use correct v2 module imports,
so that the test suite can run and provide coverage validation.

#### Acceptance Criteria
1. All test files import from existing v2 modules (main_agent_v2, rag_service_v2, tools_v2)
2. Tests execute without ModuleNotFoundError
3. Document which tests need rewriting vs simple import fixes

#### Integration Verification
- IV1: Verify CLI interface still processes queries correctly
- IV2: Confirm no changes to production code behavior
- IV3: Validate test execution time remains reasonable (<2 min total)

### Story 1.2: Remove Unused and Orphaned Modules

As a developer,
I want to safely remove all unused code modules,
so that the codebase is cleaner and maintenance burden is reduced.

#### Acceptance Criteria
1. Remove confirmed unused modules (syntax_validator, recitation_monitor, rag_optimization, rag_performance, env_validator)
2. Move debug-only modules to tests directory
3. Verify no broken imports after removal

#### Integration Verification
- IV1: Run full test suite to confirm no missing dependencies
- IV2: Test both CLI and Chainlit startup sequences
- IV3: Verify RAG service still functions with all search types

### Story 1.3: Implement Session Log Management

As a system administrator,
I want automatic session log rotation and cleanup,
so that disk space doesn't grow unbounded.

#### Acceptance Criteria
1. Implement log rotation keeping last 30 days of sessions
2. Add configuration for max log size (default 1GB)
3. Create cleanup utility for manual maintenance
4. Archive important sessions before deletion

#### Integration Verification
- IV1: Verify existing sessions remain accessible during transition
- IV2: Confirm session continuity works with rotation enabled
- IV3: Test performance with large number of session files

### Story 1.4: Complete Module Renaming and Remove v2 Suffixes

As a developer,
I want consistent module naming without version suffixes,
so that the codebase follows standard conventions.

#### Acceptance Criteria
1. Rename main_agent_v2.py → main_agent.py
2. Rename rag_service_v2.py → rag_service.py
3. Rename tools_v2.py → tools.py
4. Update all imports throughout codebase
5. Update chainlit_app_v2.py → chainlit_app.py

#### Integration Verification
- IV1: Test full query flow through both interfaces
- IV2: Verify all tool functions remain accessible
- IV3: Confirm no performance degradation from import changes

### Story 1.5: Update Documentation and Migration Guide

As a developer,
I want updated documentation reflecting the current architecture,
so that new contributors understand the system correctly.

#### Acceptance Criteria
1. Update all README references to new module names
2. Create migration guide documenting v1→v2 changes
3. Update CLAUDE.md with current patterns
4. Document test infrastructure and coverage goals
5. Add session management documentation

#### Integration Verification
- IV1: Verify all code examples in docs actually work
- IV2: Confirm API documentation matches implementation
- IV3: Test that setup instructions work for new developers

## Implementation Notes

### Story Sequencing Rationale
This sequence minimizes risk by fixing tests first (providing a safety net), then removing dead code (reducing complexity), implementing new features (session management), completing the migration (renaming), and finally updating documentation. Each story can be completed independently while maintaining system stability.

### Critical Success Factors
1. Maintain zero downtime during refactoring
2. Preserve all existing functionality
3. Improve code maintainability metrics
4. Establish sustainable development practices
5. Enable future feature development on clean foundation

### Post-Implementation Validation
- All tests passing with >80% coverage
- No orphaned or unused modules remaining
- Consistent naming throughout codebase
- Session logs under control with rotation
- Documentation fully updated and accurate