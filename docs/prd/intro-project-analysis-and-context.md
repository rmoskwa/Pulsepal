# Intro Project Analysis and Context

## Existing Project Overview

### Analysis Source
- Architecture document available at: `docs/brownfield-architecture-refactoring.md`

### Current Project State
PulsePal is an MRI sequence programming assistant built with PydanticAI, using Google Gemini 2.5 Flash as the LLM engine and Supabase for vector storage. The system has evolved from a multi-agent to single-agent architecture, leaving incomplete migration artifacts, broken tests, and redundant code modules that need cleanup.

## Documentation Analysis

### Available Documentation
- ✓ Tech Stack Documentation (from architecture doc)
- ✓ Source Tree/Architecture (detailed in architecture doc)
- ✓ Coding Standards (CLAUDE.md provides guidelines)
- ✓ API Documentation (api-docs-deploy directory)
- ✗ External API Documentation (partial)
- ✗ UX/UI Guidelines (not documented)
- ✓ Technical Debt Documentation (comprehensive in architecture doc)

## Enhancement Scope Definition

### Enhancement Type
- ✓ Bug Fix and Stability Improvements (broken tests)
- ✓ Technology Stack Upgrade (completing v1→v2 migration)
- ✓ Performance/Scalability Improvements (session management)

### Enhancement Description
Complete the unfinished v1→v2 migration, remove dead code, fix broken test infrastructure, standardize naming conventions, and implement proper session log management to reduce technical debt and improve maintainability.

### Impact Assessment
- ✓ Significant Impact (substantial existing code changes required for test fixes, module removal, and renaming)

## Goals and Background Context

### Goals
- Complete the v1→v2 migration cleanly
- Restore test suite to working condition
- Remove all unused and orphaned code modules
- Standardize module naming conventions
- Implement session log rotation strategy
- Update documentation to reflect current architecture

### Background Context
This refactoring is critical because PulsePal has accumulated significant technical debt from an incomplete architectural transition. The codebase evolved from a multi-agent to single-agent design, but the migration was never fully completed. This has left broken tests that reference non-existent modules, redundant code that adds maintenance burden, and inconsistent naming that confuses developers. The refactoring will establish a clean, maintainable foundation for future development.

## Change Log

| Change | Date | Version | Description | Author |
|--------|------|---------|-------------|--------|
| Initial PRD | 2025-01-18 | 1.0 | Refactoring PRD based on architecture analysis | John (PM) |
