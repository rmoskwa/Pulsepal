# High-Level Architecture

## Technical Summary

PulsePal is an MRI sequence programming assistant built with PydanticAI, designed to help researchers with Pulseq framework programming.

## Actual Tech Stack

| Category      | Technology           | Version      | Notes                                    |
| ------------- | -------------------- | ------------ | ---------------------------------------- |
| Runtime       | Python               | 3.10+        | WSL2 environment constraints             |
| LLM Framework | PydanticAI           | >=0.0.11     | Modern async agent framework             |
| LLM Model     | Google Gemini        | 2.5 Flash    | Primary intelligence engine              |
| Vector DB     | Supabase             | >=2.5.0      | pgvector for embeddings                  |
| Web UI        | Chainlit             | >=1.1.0      | Note: Broken on WSL2                     |
| Embeddings    | sentence-transformers| >=2.2.0      | For semantic routing                     |
| API Framework | uvicorn              | >=0.25.0     | ASGI server                              |

## Repository Structure Reality Check

- Type: Monorepo with test infrastructure issues
- Package Manager: pip/requirements.txt
- Notable Issues:
  - Tests reference non-existent v1 modules
  - Multiple versioning patterns (v2, _enhanced in cache)
  - Large conversation logs directory (500+ session files)
  - Documentation scattered across multiple locations
