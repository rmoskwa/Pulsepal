# Source Tree and Module Organization

## Project Structure (Actual with Issues)

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
