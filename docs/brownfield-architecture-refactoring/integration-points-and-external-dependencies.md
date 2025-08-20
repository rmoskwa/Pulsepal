# Integration Points and External Dependencies

## External Services

| Service        | Purpose            | Integration Status | Issues                    |
| -------------- | ------------------ | ------------------ | ------------------------- |
| Google Gemini  | LLM Provider       | ✅ Working         | None                      |
| Supabase       | Vector DB          | ✅ Working         | Project ID: mnbvsrsivuuuwbtkmumt |
| Google Embeddings | Text embeddings | ✅ Working         | Separate API key needed   |
| Chainlit       | Web UI             | ⚠️ Partial         | Broken on WSL2            |

## Internal Integration Points

- **CLI Interface**: `run_pulsepal.py` → `main_agent_v2.py`
- **Web Interface**: `chainlit_app_v2.py` → `main_agent_v2.py`
- **Session Management**: Via `dependencies.py` and `SessionManager`
- **RAG Pipeline**: `main_agent_v2.py` → `tools_v2.py` → `rag_service_v2.py`
