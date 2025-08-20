# PulsePal Test Infrastructure

## Overview

PulsePal uses a comprehensive testing strategy to ensure reliability and maintainability. Tests are organized into unit, integration, and end-to-end categories, with a focus on real service integration rather than mocking.

## Test Organization

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_main_agent.py     # Core agent logic tests
│   ├── test_rag_service.py    # RAG service unit tests
│   ├── test_dependencies.py   # Session management tests
│   └── test_tools.py          # Tool interface tests
├── integration/             # Integration tests for service interactions
│   ├── test_agent_rag.py      # Agent-RAG integration
│   ├── test_session_flow.py   # Session management flow
│   └── test_api_integration.py # External API tests
├── e2e/                     # End-to-end user workflow tests
│   ├── test_cli_workflow.py   # CLI usage scenarios
│   └── test_query_flow.py     # Full query processing
└── fixtures/                # Shared test data and utilities
    ├── sample_queries.py      # Common test queries
    └── mock_responses.py      # Mock data when needed
```

## Coverage Goals

### Target Coverage
- **80% overall coverage** for the pulsepal package
- **100% coverage** for critical paths:
  - Agent tool functions
  - RAG search methods
  - Session management
  - Error handling paths

### Current Coverage Status
Run `pytest --cov=pulsepal` to see current coverage metrics.

## Running Tests

### Basic Test Commands

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=pulsepal --cov-report=html

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_main_agent.py

# Run specific test function
pytest tests/unit/test_main_agent.py::test_agent_initialization
```

### Test Categories

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run only end-to-end tests
pytest tests/e2e/

# Run tests in parallel (requires pytest-xdist)
pytest -n auto
```

### Test Options

```bash
# Stop on first failure
pytest -x

# Run only last failed tests
pytest --lf

# Run tests matching a pattern
pytest -k "rag_search"

# Show local variables in tracebacks
pytest -l

# Generate HTML coverage report
pytest --cov=pulsepal --cov-report=html
# View report at htmlcov/index.html
```

## Test Fixtures

Common fixtures are available in `tests/fixtures/`:

```python
# Example fixture usage
@pytest.fixture
def sample_query():
    return "How do I create a gradient echo sequence?"

@pytest.fixture
async def rag_service():
    from pulsepal.rag_service import RAGService
    return RAGService()

@pytest.fixture
def session_manager():
    from pulsepal.dependencies import SessionManager
    return SessionManager()
```

## Writing Tests

### Test Naming Conventions

- Use descriptive names: `test_<what>_<condition>_<expected_result>`
- Examples:
  - `test_rag_search_with_valid_query_returns_results`
  - `test_session_creation_generates_unique_id`
  - `test_agent_response_includes_code_examples`

### Test Structure

```python
import pytest
from pulsepal.main_agent import pulsepal_agent

class TestMainAgent:
    """Group related tests in classes."""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initializes correctly."""
        # Arrange
        query = "test query"
        
        # Act
        result = await pulsepal_agent.run(query)
        
        # Assert
        assert result is not None
        assert isinstance(result.data, str)
    
    @pytest.mark.parametrize("query,expected", [
        ("gradient echo", True),
        ("spin echo", True),
        ("invalid", False),
    ])
    async def test_query_validation(self, query, expected):
        """Test query validation with multiple inputs."""
        # Test implementation
```

## Environment Setup

### Required Environment Variables

For integration tests, create a `.env.test` file:

```bash
# Test environment variables
GOOGLE_API_KEY=your-test-api-key
SUPABASE_URL=your-test-supabase-url
SUPABASE_KEY=your-test-supabase-key
LLM_MODEL=gemini-2.5-flash
TESTING=true
```

### Test Database

Integration tests use the same Supabase instance but can be configured to use a test schema:

```python
# In test setup
os.environ["SUPABASE_SCHEMA"] = "test"
```

## Continuous Integration

Tests run automatically on:
- Pull requests
- Commits to main branch
- Nightly builds

CI configuration in `.github/workflows/tests.yml`

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure `pulsepal` package is installed: `pip install -e .`
   - Check PYTHONPATH includes project root

2. **Async Test Failures**
   - Use `@pytest.mark.asyncio` decorator
   - Install pytest-asyncio: `pip install pytest-asyncio`

3. **API Rate Limits**
   - Use fixtures to cache API responses
   - Implement retry logic with exponential backoff

4. **Session Test Failures**
   - Clean up test sessions after each test
   - Use unique session IDs for parallel tests

### Debug Commands

```bash
# Run tests with debugging
pytest --pdb  # Drop into debugger on failure

# Show print statements
pytest -s

# Generate detailed failure report
pytest --tb=long

# Check test collection without running
pytest --collect-only
```

## Best Practices

1. **Prefer Integration Tests** - Test with real services when possible
2. **Use Fixtures** - Share common setup across tests
3. **Test Error Paths** - Ensure proper error handling
4. **Keep Tests Fast** - Use mocks only for slow external calls
5. **Test One Thing** - Each test should verify a single behavior
6. **Clean Up** - Always clean up resources (sessions, files, etc.)

## Monitoring Test Health

### Metrics to Track
- Test execution time
- Flaky test frequency
- Coverage trends
- Test failure patterns

### Commands for Analysis

```bash
# Find slow tests
pytest --durations=10

# Generate test timing report
pytest --junit-xml=test-results.xml

# Check for test dependencies
pytest --randomly-dont-shuffle
```

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass locally
3. Maintain or improve coverage
4. Update this README if test structure changes

For questions about testing, contact: rmoskwa@wisc.edu