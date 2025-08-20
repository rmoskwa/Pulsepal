# Epic 1: PulsePal Technical Debt Elimination and v2 Migration Completion

**Epic Goal**: Complete the v1→v2 migration, eliminate technical debt, and establish a clean, maintainable codebase with working tests and proper session management

**Integration Requirements**: All changes must maintain backward compatibility with existing interfaces; each story must verify existing functionality remains intact

## Story 1.1: Fix Test Infrastructure and Update Imports

As a developer,
I want to update all test files to use correct v2 module imports,
so that the test suite can run and provide coverage validation.

### Acceptance Criteria
1. All test files import from existing v2 modules (main_agent_v2, rag_service_v2, tools_v2)
2. Tests execute without ModuleNotFoundError
3. Document which tests need rewriting vs simple import fixes

### Integration Verification
- IV1: Verify CLI interface still processes queries correctly
- IV2: Confirm no changes to production code behavior
- IV3: Validate test execution time remains reasonable (<2 min total)

## Story 1.2: Remove Unused and Orphaned Modules

As a developer,
I want to safely remove all unused code modules,
so that the codebase is cleaner and maintenance burden is reduced.

### Acceptance Criteria
1. Remove confirmed unused modules (syntax_validator, recitation_monitor, rag_optimization, rag_performance, env_validator)
2. Move debug-only modules to tests directory
3. Verify no broken imports after removal

### Integration Verification
- IV1: Run full test suite to confirm no missing dependencies
- IV2: Test both CLI and Chainlit startup sequences
- IV3: Verify RAG service still functions with all search types

## Story 1.3: Implement Session Log Management

As a system administrator,
I want automatic session log rotation and cleanup,
so that disk space doesn't grow unbounded.

### Acceptance Criteria
1. Implement log rotation keeping last 30 days of sessions
2. Add configuration for max log size (default 1GB)
3. Create cleanup utility for manual maintenance
4. Archive important sessions before deletion

### Integration Verification
- IV1: Verify existing sessions remain accessible during transition
- IV2: Confirm session continuity works with rotation enabled
- IV3: Test performance with large number of session files

## Story 1.4: Complete Module Renaming and Remove v2 Suffixes

As a developer,
I want consistent module naming without version suffixes,
so that the codebase follows standard conventions.

### Acceptance Criteria
1. Rename main_agent_v2.py → main_agent.py
2. Rename rag_service_v2.py → rag_service.py
3. Rename tools_v2.py → tools.py
4. Update all imports throughout codebase
5. Update chainlit_app_v2.py → chainlit_app.py

### Integration Verification
- IV1: Test full query flow through both interfaces
- IV2: Verify all tool functions remain accessible
- IV3: Confirm no performance degradation from import changes

## Story 1.5: Update Documentation and Migration Guide

As a developer,
I want updated documentation reflecting the current architecture,
so that new contributors understand the system correctly.

### Acceptance Criteria
1. Update all README references to new module names
2. Create migration guide documenting v1→v2 changes
3. Update CLAUDE.md with current patterns
4. Document test infrastructure and coverage goals
5. Add session management documentation

### Integration Verification
- IV1: Verify all code examples in docs actually work
- IV2: Confirm API documentation matches implementation
- IV3: Test that setup instructions work for new developers
