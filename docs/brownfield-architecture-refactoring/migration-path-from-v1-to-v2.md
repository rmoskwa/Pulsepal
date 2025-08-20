# Migration Path from V1 to V2

## What Changed
- `main_agent.py` → `main_agent_v2.py`
- `rag_service.py` → `rag_service_v2.py`
- `tools.py` → `tools_v2.py`
- Multi-agent architecture → Single intelligent agent
- Complex routing → Simplified semantic routing

## What Broke
- All tests importing old module names
- Function discovery features referenced in tests
- Any documentation referencing old architecture
