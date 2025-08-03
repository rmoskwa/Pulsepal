# Enhanced Product Requirement Prompt (PRP): Pulsepal Intelligence Refactor

## 🎯 Project Context
You are refactoring Pulsepal, a PydanticAI-based assistant for MRI Pulseq programming. The current system forces constant tool usage and has redundant components. We're transforming it into an intelligent LLM that selectively uses a Pulseq knowledge base only when needed.

## ⚙️ Configuration Updates

### LLM Model Selection
**Use `gemini-2.5-flash`** instead of flash-lite for enhanced reasoning capabilities:
```python
# In settings.py
llm_model: str = Field(
    default="gemini-2.5-flash",  # Using full flash for debugging support
    description="Gemini model with reasoning capabilities"
)
```

**Rationale**: 30-40% of Pulsepal queries will be debugging-related, requiring reasoning capabilities.

### Embedding Model Configuration
**Note**: Project uses **local embeddings** (BGE model) - no embedding API costs:
```python
# Already configured in settings.py
bge_model_path: str = Field(
    default="/path/to/local/BGE/model",
    description="Local BGE embedding model - no API costs"
)
```

## 📁 Current Problem
- The agent ALWAYS searches the knowledge base, even for general MRI questions
- Has a redundant "MRI Expert" agent (LLMs already know MRI physics)
- Uses 4 separate tools inefficiently
- Slow responses due to unnecessary searches
- Acts like a search engine rather than an intelligent assistant

## ✅ Desired Outcome
Transform Pulsepal into a system that:
- Uses the LLM's inherent knowledge for 90% of queries
- Only searches Pulseq documentation for specific API details
- Responds 2-3 seconds faster for general MRI/programming questions
- Seamlessly integrates search results when truly needed
- Behaves like "Claude/GPT-4 that has read all Pulseq documentation"

## Key Benefits (with Local Embeddings):
Since you're using local embeddings (no API costs), the main improvements are:
- **Speed**: 2-3 second reduction for 90% of queries
- **Quality**: No forced integration of irrelevant search results
- **Intelligence**: Natural responses that know when searching adds value
- **Database Load**: 80% fewer Supabase queries
- **User Experience**: No more "searching..." for obvious questions

## 📋 Implementation Requirements

### Phase 1: Remove Redundant Components

1. **Delete MRI Expert Agent Files**
   - Delete `pulsepal/mri_expert_agent.py`
   - Delete `pulsepal/tests/test_agent_delegation.py`
   - Remove all imports and references to MRI Expert throughout the codebase
   - Search for: `mri_expert`, `MRIExpert`, `delegate_to_mri_expert`

2. **Clean Dependencies**
   - In `pulsepal/dependencies.py`:
     - Remove the `MRIExpertDependencies` class
     - Remove any MRI Expert related imports
     - Keep `PulsePalDependencies` and session management intact

### Phase 2: Rewrite System Prompt

3. **Update main_agent.py System Prompt**
   Replace the entire `PULSEPAL_SYSTEM_PROMPT` with:
   ```python
   PULSEPAL_SYSTEM_PROMPT = """You are Pulsepal, an advanced AI assistant specializing in Pulseq MRI sequence programming.

   You are a powerful language model with comprehensive built-in knowledge of MRI physics, programming, and scientific computing, enhanced with access to specialized Pulseq documentation when specifically needed.

   ## Core Operating Principle
   You are like an expert MRI researcher who has instant access to Pulseq documentation. Use your inherent knowledge for most queries, and only search the Pulseq knowledge base for specific implementation details.

   ## Decision Framework

   ### Use YOUR KNOWLEDGE (no tools) for:
   - MRI physics concepts (T1/T2 relaxation, k-space, gradients, pulse sequences)
   - Programming concepts and syntax (any language)
   - Mathematical calculations and formulas
   - Standard sequence types and their principles
   - General debugging and optimization strategies
   - Safety considerations (SAR, PNS, etc.)

   ### Search Pulseq knowledge base ONLY for:
   - Specific Pulseq function signatures (e.g., exact parameters for mr.makeGaussPulse)
   - Implementation examples from Pulseq repositories
   - Community-contributed sequences (MOLLI, SMS-EPI, etc.)
   - Version-specific features or compatibility
   - Pulseq-specific optimization techniques
   - Undocumented tricks from real implementations

   ## Debugging Support (Enhanced for gemini-2.5-flash)
   When debugging Pulseq code:
   1. First analyze the code using your knowledge
   2. Search ONLY if the error involves Pulseq-specific functions
   3. Use reasoning to trace through logic and identify issues
   4. Provide step-by-step debugging guidance

   ## Response Strategy
   1. Analyze if the query needs Pulseq-specific information
   2. If general knowledge suffices, respond immediately
   3. If Pulseq details needed, search selectively and integrate naturally
   4. Never mention "searching" or "checking documentation" unless relevant
   5. Present all information as your knowledge

   ## Language Support
   - Default to MATLAB for code examples unless specified otherwise
   - Support: MATLAB, Python (pypulseq), Octave, C/C++, Julia
   - Detect user's preferred language from context

   ## Examples of Decision Making
   - "What is a spin echo?" → Use knowledge (general MRI)
   - "How to use mr.makeBlockPulse?" → Search (Pulseq-specific)
   - "Explain k-space" → Use knowledge (general concept)
   - "Show MOLLI implementation" → Search (specific sequence)
   - "Debug this code" → Analyze first, search only if Pulseq functions involved
   - "Why does my sequence crash?" → Use reasoning, search if needed

   Remember: You are an intelligent assistant enhanced with Pulseq knowledge, not a search interface."""
   ```

### Phase 3: Simplify and Optimize Tools

4. **Create New Unified Tool System**
   Replace the entire `pulsepal/tools.py` with the simplified version (see original PRP above)

5. **Remove old tool references**
   - Remove `delegate_to_mri_expert` tool
   - Remove `get_available_sources` tool  
   - Remove separate `perform_rag_query` and `search_code_examples` tools

### Phase 4: Update Agent Logic

6. **Simplify main_agent.py run function** (see original PRP above)

7. **Remove forced tool usage from imports**

### Phase 5: Update UI Components

8. **Update chainlit_app.py** (see original PRP above)

9. **Update Welcome Message** (see original PRP above)

### Phase 6: Testing and Validation

10. **Create Intelligence Test Suite**
    Add debugging-specific test cases:
    ```python
    """Test intelligent tool usage patterns."""
    
    test_cases = [
        # Should NOT trigger search
        {"query": "What is T1 relaxation?", "should_search": False},
        {"query": "Explain gradient echo vs spin echo", "should_search": False},
        {"query": "How does k-space work?", "should_search": False},
        {"query": "What causes image artifacts?", "should_search": False},
        {"query": "Debug: undefined variable error", "should_search": False},
        {"query": "Why does my loop run forever?", "should_search": False},
        
        # SHOULD trigger search
        {"query": "How to use mr.makeGaussPulse?", "should_search": True},
        {"query": "Show MOLLI sequence code", "should_search": True},
        {"query": "pypulseq calcDuration parameters", "should_search": True},
        {"query": "Pulseq v1.5.0 new features", "should_search": True},
        {"query": "Debug: mr.addBlock timing error", "should_search": True},
    ]
    ```

11. **Add Performance Metrics** (see original PRP above)

### Phase 7: Documentation Updates

12. **Update README.md and Documentation** (see original PRP above)

## 🤖 Claude Code Subagent Usage Instructions

### During Implementation:

#### 1. **After Phase 1 (Removing MRI Expert)**
Use `/agents critical-code-reviewer` to:
- Review the dependency cleanup
- Ensure no orphaned references remain
- Validate that session management is intact

#### 2. **After Phase 3 (Tool Simplification)**  
Use `/agents pydantic-ai-test-suite` to:
- Create comprehensive tests for the new unified tool
- Test intelligent decision-making logic
- Validate search filtering works correctly

#### 3. **After Phase 5 (UI Updates)**
Use `/agents pulsepal-docs-generator` to:
- Generate updated user documentation
- Create examples of the new intelligent behavior
- Document the performance improvements

#### 4. **If Issues Arise**
Use `/agents system-debugger` to:
- Debug any MCP server connection issues
- Analyze why certain queries trigger unwanted searches
- Investigate performance bottlenecks

## 📊 Success Metrics

### Before Refactoring (Current Performance):
- **Response time**: 4-7 seconds for ALL queries (LLM + forced search)
- **Search rate**: ~100% of queries (searches even for "What is T1?")
- **Database load**: High - every query hits Supabase
- **Quality**: Often includes irrelevant search results

### After Refactoring (Realistic Targets):
- **Response times**:
  - General queries (90%): 2-4 seconds (just LLM, no search)
  - Pulseq-specific (10%): 3-6 seconds (LLM + search when needed)
  - Complex debugging: 4-7 seconds (reasoning + conditional search)
- **Search rate**: <20% of queries
- **Database load**: 80% reduction in Supabase queries
- **Quality**: Clean, direct responses without forced search results

### Debugging-Specific Metrics:
- **Simple syntax errors**: 0 searches needed, 2-4 seconds
- **Complex debugging**: 1 search maximum, 4-7 seconds
- **Reasoning utilization**: Full use of Gemini 2.5 Flash capabilities

### Cost Impact:
- **LLM costs**: Slight reduction (fewer tokens without search results)
- **Embedding costs**: N/A (using local BGE model)
- **Database costs**: Reduced Supabase queries

## 🧪 Testing Checklist

- [ ] General MRI questions answered without search
- [ ] Pulseq-specific queries trigger appropriate searches
- [ ] Debugging works effectively with reasoning
- [ ] Response quality maintained or improved
- [ ] No references to removed MRI Expert
- [ ] UI properly updated
- [ ] Session management still works
- [ ] Language detection still functional
- [ ] Performance metrics show improvement

## 📝 Implementation Order

1. **Start with Phase 1**: Remove MRI Expert completely
2. **Test after Phase 1**: Ensure nothing breaks
3. **Implement Phase 2-3**: New prompt and tools
4. **Test intelligence**: Verify decision-making works
5. **Update UI (Phase 5)**: Polish user experience
6. **Run full test suite**: Validate everything works
7. **Document changes**: Update all documentation

## ⚠️ Critical Notes

1. **Preserve Working Features**:
   - Session management must continue working
   - Language detection should remain functional
   - Conversation history must be maintained

2. **Gemini 2.5 Flash Configuration**:
   ```python
   # Ensure .env has:
   LLM_MODEL=gemini-2.5-flash
   # NOT gemini-2.5-flash-lite
   ```

3. **Local Embeddings Configuration**:
   ```python
   # Already configured - no changes needed:
   BGE_MODEL_PATH=/path/to/local/BGE/model
   # This means NO embedding API costs
   ```

4. **Debugging Focus**:
   - Since 30-40% of queries are debugging-related
   - Ensure reasoning capabilities are fully utilized
   - Test complex debugging scenarios thoroughly

5. **Performance Expectations**:
   - LLMs take time - 2-4 seconds is normal for Gemini 2.5 Flash
   - The win is avoiding unnecessary 1-2 second search overhead
   - Focus on quality and intelligence, not sub-second responses

## 🔄 Rollback Plan

If issues arise:
1. Git commit current state before changes: `git commit -am "Backup before intelligence refactor"`
2. Create branch for changes: `git checkout -b intelligent-refactor`
3. Test each phase independently
4. Keep original files in `.backup/` directory
5. Document any issues encountered

## 📈 Expected Impact

- **User Experience**: 2-3 second faster responses for 90% of queries
- **System Performance**: 80% reduction in Supabase database queries
- **Response Quality**: Better answers without forced search integration
- **Debugging**: Enhanced capability with Gemini 2.5 Flash reasoning
- **Maintainability**: 50% less code complexity without MRI Expert delegation
- **Intelligence**: Natural responses that know when to search vs. when to use built-in knowledge

## 🎯 Final Validation

After implementation, run this validation script:
```python
# validation.py
import asyncio
import time
from pulsepal.main_agent import run_pulsepal

async def validate():
    test_cases = [
        ("What is T1 relaxation?", 2.0, 4.0),  # Should be 2-4 seconds (no search)
        ("How to use mr.makeGaussPulse?", 3.0, 6.0),  # Should be 3-6 seconds (with search)
        ("Debug: undefined variable rf_pulse", 2.0, 5.0),  # Should be 2-5 seconds (reasoning)
    ]
    
    for query, min_expected, max_expected in test_cases:
        start = time.time()
        _, response = await run_pulsepal(query)
        duration = time.time() - start
        
        if min_expected <= duration <= max_expected:
            status = "✅ PASS"
        else:
            status = "⚠️ OUTSIDE EXPECTED RANGE"
            
        print(f"{status} | Query: {query[:30]}...")
        print(f"  Time: {duration:.2f}s (Expected: {min_expected}-{max_expected}s)\n")

asyncio.run(validate())
```

Success = All queries complete within expected time ranges!