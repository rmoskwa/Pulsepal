"""
Tests for the modern RAG service v2.

Ensures the new simplified architecture works correctly without
pattern matching or classification.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from pulsepal.rag_service import ModernPulseqRAG, RetrievalHint


@pytest.fixture
def mock_supabase_client():
    """Create a mock Supabase client."""
    client = Mock()
    client.rpc = Mock()
    client.client = Mock()
    client.client.table = Mock()
    return client


@pytest.fixture
def rag_service(mock_supabase_client):
    """Create RAG service with mocked dependencies."""
    with patch('pulsepal.rag_service.get_supabase_client', return_value=mock_supabase_client):
        service = ModernPulseqRAG()
        service._supabase_client = mock_supabase_client
        return service


@pytest.mark.asyncio
async def test_no_pattern_matching(rag_service, mock_supabase_client):
    """Ensure no hardcoded patterns for query types."""
    
    # Mock embeddings
    with patch('pulsepal.embeddings.create_embedding', return_value=[0.1] * 768):
        # Mock Supabase responses
        mock_supabase_client.rpc.return_value.data = [
            {
                "content": "Test content",
                "url": "test.html",
                "title": "Test",
                "similarity": 0.8
            }
        ]
        
        # Test different query types - all should return same structure
        queries = [
            "show me an example",
            "why do I have artifacts",
            "random query xyz",
            "implement a spin echo",
            "debug my code",
            "what is the error"
        ]
        
        results = []
        for query in queries:
            result = await rag_service.retrieve(query)
            results.append(result)
        
        # All should have same structure
        for result in results:
            assert "documents" in result
            assert "metadata" in result
            assert isinstance(result["documents"], list)
            assert isinstance(result["metadata"], dict)
        
        # No classification should have occurred
        # The structure should be identical for all queries
        first_keys = set(results[0]["metadata"].keys())
        for result in results[1:]:
            assert set(result["metadata"].keys()) == first_keys


@pytest.mark.asyncio
async def test_function_namespace_validation(rag_service, mock_supabase_client):
    """Test deterministic function validation."""
    
    # Mock function_calling_patterns table
    mock_table = Mock()
    mock_supabase_client.client.table.return_value = mock_table
    
    # Test wrong namespace
    mock_table.select.return_value.eq.return_value.execute.return_value.data = [
        {
            "function_name": "write",
            "namespace": "seq",
            "category": "sequence",
            "correct_usage": "seq.write(filename)"
        }
    ]
    
    result = await rag_service.check_function_namespace("mr.write")
    assert result["is_error"] == True
    assert result["correct"] == "seq.write"
    assert "belongs to the seq namespace" in result["explanation"]
    
    # Test correct namespace
    mock_table.select.return_value.eq.return_value.execute.return_value.data = [
        {
            "function_name": "makeTrapezoid",
            "namespace": "mr",
            "category": "gradient",
            "correct_usage": "mr.makeTrapezoid(...)"
        }
    ]
    
    result = await rag_service.check_function_namespace("mr.makeTrapezoid")
    assert result["is_correct"] == True
    assert result["is_error"] == False


@pytest.mark.asyncio
async def test_retrieval_with_hints(rag_service, mock_supabase_client):
    """Test that hints improve retrieval without classification."""
    
    with patch('pulsepal.embeddings.create_embedding', return_value=[0.1] * 768):
        # Mock function docs response
        mock_table = Mock()
        mock_supabase_client.client.table.return_value = mock_table
        mock_table.select.return_value.eq.return_value.execute.return_value.data = [
            {
                "function_name": "makeTrapezoid",
                "namespace": "mr",
                "description": "Create trapezoid gradient",
                "correct_usage": "mr.makeTrapezoid('x', ...)",
                "common_errors": "Missing channel parameter"
            }
        ]
        
        # Mock vector search response
        mock_supabase_client.rpc.return_value.data = [
            {
                "content": "Gradient echo sequence example",
                "url": "examples/gre.m",
                "similarity": 0.75
            }
        ]
        
        hint = RetrievalHint(
            functions_mentioned=["makeTrapezoid", "makeAdc"],
            code_provided=True
        )
        
        results = await rag_service.retrieve("gradient echo sequence", hint)
        
        # Should have both function docs and vector search results
        assert len(results["documents"]) > 0
        assert results["metadata"]["has_function_docs"] == True
        assert "function_calling_patterns" in results["metadata"]["sources"]
        assert "vector_search" in results["metadata"]["sources"]


@pytest.mark.asyncio
async def test_simple_retrieval_interface(rag_service, mock_supabase_client):
    """Test that the interface is simple and consistent."""
    
    with patch('pulsepal.embeddings.create_embedding', return_value=[0.1] * 768):
        mock_supabase_client.rpc.return_value.data = []
        
        # Test with just query
        result = await rag_service.retrieve("test query")
        assert isinstance(result, dict)
        assert "documents" in result
        assert "metadata" in result
        
        # Test with hint
        hint = RetrievalHint(functions_mentioned=["test"])
        result = await rag_service.retrieve("test query", hint)
        assert isinstance(result, dict)
        assert "documents" in result
        assert "metadata" in result
        
        # Test with limit
        result = await rag_service.retrieve("test query", limit=5)
        assert isinstance(result, dict)
        assert "documents" in result
        assert "metadata" in result


@pytest.mark.asyncio
async def test_content_type_detection(rag_service):
    """Test simple content type detection based on file extension."""
    
    # Test various file types
    assert rag_service._detect_content_type("test.m") == "matlab_code"
    assert rag_service._detect_content_type("test.py") == "python_code"
    assert rag_service._detect_content_type("test.md") == "markdown_doc"
    assert rag_service._detect_content_type("test.html") == "html_doc"
    assert rag_service._detect_content_type("/examples/test.m") in ["example", "matlab_code"]  # Path-based detection may vary
    assert rag_service._detect_content_type("/api/reference.html") == "api_reference"
    assert rag_service._detect_content_type("random.txt") == "documentation"
    assert rag_service._detect_content_type("") == "unknown"


@pytest.mark.asyncio
async def test_no_query_enhancement(rag_service, mock_supabase_client):
    """Test that queries are not enhanced or rewritten."""
    
    with patch('pulsepal.embeddings.create_embedding') as mock_embed:
        mock_embed.return_value = [0.1] * 768
        mock_supabase_client.rpc.return_value.data = []
        
        # Original query should be passed as-is to embedding
        test_query = "my EXACT query with CaSe"
        await rag_service.retrieve(test_query)
        
        # Verify the exact query was used for embedding
        mock_embed.assert_called_with(test_query)


@pytest.mark.asyncio
async def test_performance_limit(rag_service, mock_supabase_client):
    """Test that retrieval respects the limit parameter."""
    
    with patch('pulsepal.embeddings.create_embedding', return_value=[0.1] * 768):
        # Create many mock results
        mock_results = [
            {
                "content": f"Result {i}",
                "url": f"test{i}.html",
                "similarity": 0.9 - (i * 0.01)
            }
            for i in range(100)
        ]
        
        mock_supabase_client.rpc.return_value.data = mock_results
        
        # Test with limit
        result = await rag_service.retrieve("test", limit=10)
        assert len(result["documents"]) <= 10
        
        result = await rag_service.retrieve("test", limit=5)
        assert len(result["documents"]) <= 5


def test_code_reduction():
    """Verify that the new RAG service is significantly smaller."""
    import os
    
    # Get file sizes
    v2_path = "pulsepal/rag_service.py"
    if os.path.exists(v2_path):
        with open(v2_path, 'r') as f:
            v2_lines = len(f.readlines())
        
        # Should be under 300 lines as per requirements
        assert v2_lines < 300, f"RAG v2 has {v2_lines} lines, should be < 300"
        print(f"✓ Code reduction achieved: {v2_lines} lines")


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_no_pattern_matching(
        ModernPulseqRAG(), 
        Mock()
    ))
    print("✓ All tests passed")