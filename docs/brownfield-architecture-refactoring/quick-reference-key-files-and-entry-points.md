# Quick Reference - Key Files and Entry Points

## Critical Active Files (Currently in Use)
- **Main CLI Entry**: `run_pulsepal.py` - CLI interface using v2 components
- **Web UI Entry**: `chainlit_app_v2.py` - Chainlit web interface
- **Core Agent**: `pulsepal/main_agent_v2.py` - Single intelligent agent
- **RAG Service**: `pulsepal/rag_service_v2.py` - Modern RAG with hybrid search
- **Tools Interface**: `pulsepal/tools_v2.py` - Unified tool interface
- **Configuration**: `pulsepal/settings.py` - Environment configuration
- **Dependencies**: `pulsepal/dependencies.py` - Session management

## Orphaned/Broken References (Need Removal)
- **Non-existent**: `pulsepal/main_agent.py` - Referenced by tests but doesn't exist
- **Non-existent**: `pulsepal/rag_service.py` - Referenced by tests but doesn't exist
- **Non-existent**: `pulsepal/tools.py` - Referenced by tests but doesn't exist
