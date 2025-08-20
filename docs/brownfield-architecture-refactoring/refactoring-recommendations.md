# Refactoring Recommendations

## Immediate Actions Needed

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

## Module-Specific Refactoring Needs

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
