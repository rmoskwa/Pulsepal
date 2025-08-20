"""
Test suite for RAG enhancements.

Test cases:
1. Function lookup returns correct calling pattern
2. Example request returns runnable code
3. Debug request identifies and fixes errors
4. Tutorial request returns educational content
5. MATLAB is default, Python only when specified
6. Class methods vs regular functions handled correctly
7. Notebook processing works for both intents
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch
from pulsepal.rag_service import ModernPulseqRAG as RAGService


class TestRAGEnhancement:
    """Test suite for enhanced RAG functionality."""
    
    @pytest.fixture
    def rag_service(self):
        """Create RAG service instance for testing."""
        with patch('pulsepal.rag_service.get_supabase_client'):
            service = RAGService()
            # Mock the supabase client
            service._supabase_client = MagicMock()
            return service
    
    def test_query_intent_classification(self, rag_service):
        """Test that query intent is correctly classified."""
        
        # Test function lookup detection
        result = rag_service.classify_query_intent("How do I use mr.makeTrapezoid?")
        assert result['intent'] == 'function_lookup'
        assert result['confidence'] >= 0.7
        
        # Test example request detection
        result = rag_service.classify_query_intent("Show me a spin echo sequence example")
        assert result['intent'] == 'example_request'
        
        # Test debug request detection
        result = rag_service.classify_query_intent("I'm getting 'undefined function mr.write' error")
        assert result['intent'] == 'debug_request'
        assert result['confidence'] == 1.0  # Special pattern override
        
        # Test tutorial request detection
        result = rag_service.classify_query_intent("Tutorial on creating gradient echo sequence")
        assert result['intent'] == 'tutorial_request'
        
        # Test concept question detection
        result = rag_service.classify_query_intent("What is k-space?")
        assert result['intent'] == 'concept_question'
    
    def test_language_detection(self, rag_service):
        """Test that language preference is correctly detected."""
        
        # Default should be MATLAB
        result = rag_service.classify_query_intent("Create a spin echo sequence")
        assert result['language'] == 'matlab'
        
        # Explicit Python request
        result = rag_service.classify_query_intent("Create an EPI sequence in pypulseq")
        assert result['language'] == 'python'
        
        # Explicit MATLAB request
        result = rag_service.classify_query_intent("Show me MATLAB code for gradient echo")
        assert result['language'] == 'matlab'
    
    @pytest.mark.asyncio
    async def test_enhanced_api_search(self, rag_service):
        """Test enhanced API function search with function_calling_patterns view."""
        # Mock the database response
        mock_results = [{
            'function_name': 'write',
            'calling_pattern': 'seq.write(filename)',
            'usage_instruction': 'This is a Sequence class method. First create: seq = mr.Sequence();',
            'parameters': 'filename: string',
            'description': 'Write sequence to file',
            'is_class_method': True,
            'language': 'matlab'
        }]
        
        rag_service.supabase_client.client.from_().select().eq().limit().execute = MagicMock(
            return_value=MagicMock(data=mock_results)
        )
        
        result = await rag_service.search_api_functions_enhanced("write", "matlab", 5)
        
        assert "seq.write" in result
        assert "Class Method" in result
        assert "First create the sequence object" in result
    
    def test_validate_pulseq_code(self, rag_service):
        """Test code validation for common errors."""
        
        code = """
        seq = mr.Sequence();
        rf = mr.makeSincPulse(500e-6, sys);
        mr.write('test.seq');  % Error: should be seq.write
        seq.addBlock(rf);
        """
        
        validation = rag_service.validate_pulseq_code(code, "matlab")
        
        # Should detect the mr.write error
        assert len(validation['errors']) > 0
        assert any('mr.write' in str(error) for error in validation['errors'])
        
        # Should have valid functions too
        assert 'mr.makeSincPulse' in str(validation['valid_functions'])
    
    def test_process_notebook_content(self, rag_service):
        """Test notebook processing for different intents."""
        
        notebook_json = """{
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["# Spin Echo Sequence Tutorial"]
                },
                {
                    "cell_type": "code",
                    "source": ["seq = mr.Sequence();\\nrf = mr.makeSincPulse(500e-6);"]
                }
            ]
        }"""
        
        # Test example request processing (code only)
        result = rag_service.process_notebook_content(notebook_json, "example_request")
        assert "Extracted from Jupyter notebook" in result
        assert "seq = mr.Sequence();" in result
        assert "Tutorial" not in result  # Markdown should be removed
        
        # Test tutorial request processing (preserves structure)
        result = rag_service.process_notebook_content(notebook_json, "tutorial_request")
        assert "=== Explanation ===" in result
        assert "Spin Echo Sequence Tutorial" in result
        assert "=== Code Block" in result
    
    def test_adaptive_result_formatting(self, rag_service):
        """Test that results are formatted based on intent."""
        
        mock_results = [{
            'content': 'seq = mr.Sequence();\nrf = mr.makeSincPulse(500e-6);',
            'summary': 'Basic spin echo sequence',
            'url': 'example.com/spinecho.m'
        }]
        
        # Test example request formatting
        result = rag_service.format_results_adaptive(mock_results, 'example_request', 'spin echo')
        assert '```matlab' in result
        assert 'seq = mr.Sequence()' in result
        
        # Test debug request formatting
        result = rag_service.format_results_adaptive([], 'debug_request', 'mr.write error')
        assert 'Issue Found' in result or 'Debugging Help' in result
        
        # Test concept question formatting
        result = rag_service.format_results_adaptive(mock_results, 'concept_question', 'spin echo')
        assert 'Concept:' in result
    
    def test_code_implementation_search(self, rag_service):
        """Test searching for code implementations in crawled_pages."""
        
        mock_results = [{
            'content': 'seq = mr.Sequence();\n% Full sequence code\nseq.write("test.seq");',
            'summary': 'Complete spin echo implementation',
            'url': 'github.com/example/spinecho.m',
            'metadata': {'file_extension': '.m', 'language': 'matlab'}
        }]
        
        rag_service.supabase_client.client.from_().select().or_().or_().limit().execute = MagicMock(
            return_value=MagicMock(data=mock_results)
        )
        
        result = rag_service.search_code_implementations("spin echo", "matlab", 5)
        
        assert "Code Implementations" in result
        assert "MATLAB" in result
        assert "seq.write" in result  # Should prioritize complete sequences
    
    def test_extract_functions_used(self, rag_service):
        """Test extraction of Pulseq functions from code."""
        
        code = """
        seq = mr.Sequence();
        rf = mr.makeSincPulse(500e-6, sys);
        gx = mr.makeTrapezoid('x', sys);
        seq.addBlock(rf, gx);
        seq.write('test.seq');
        """
        
        functions = rag_service._extract_functions_used(code)
        
        assert 'mr.makeSincPulse' in functions
        assert 'mr.makeTrapezoid' in functions
        assert 'seq.addBlock' in functions
        assert 'seq.write' in functions
    
    @pytest.mark.asyncio
    async def test_enhanced_perform_rag_query(self, rag_service):
        """Test the enhanced main RAG query with intelligent routing."""
        
        # Mock classify_query_intent
        rag_service.classify_query_intent = MagicMock(return_value={
            'intent': 'function_lookup',
            'confidence': 0.9,
            'language': 'matlab',
            'search_strategy': 'api_enhanced'
        })
        
        # Mock the enhanced API search
        with patch.object(rag_service, 'search_api_functions_enhanced', 
                         return_value="## Function Reference\n**Calling Pattern:** `mr.makeTrapezoid()`"):
            
            result = await rag_service.perform_rag_query("How to use makeTrapezoid", "auto", 5)
            
            assert "Function Reference" in result
            assert "makeTrapezoid" in result
    
    def test_common_error_detection(self, rag_service):
        """Test detection of common Pulseq errors."""
        
        # Test mr.write error detection
        validation = rag_service.validate_pulseq_code("mr.write('test.seq');", "matlab")
        assert len(validation['errors']) > 0
        assert any('Sequence class method' in error['error'] for error in validation['errors'])
        
        # Test suggestions for missing sequence creation
        validation = rag_service.validate_pulseq_code("seq.addBlock(rf);", "matlab")
        assert any('seq = mr.Sequence()' in suggestion for suggestion in validation['suggestions'])


# Test queries that should be handled differently
TEST_QUERIES = [
    ("How do I use mr.makeTrapezoid?", "function_lookup", "matlab"),
    ("Show me a spin echo sequence example", "example_request", "matlab"),
    ("I'm getting 'undefined function mr.write' error", "debug_request", "matlab"),
    ("Tutorial on creating gradient echo sequence", "tutorial_request", "matlab"),
    ("Create an EPI sequence in pypulseq", "example_request", "python"),
]


@pytest.mark.parametrize("query,expected_intent,expected_language", TEST_QUERIES)
def test_query_classification_parametrized(query, expected_intent, expected_language):
    """Parametrized test for query classification."""
    with patch('pulsepal.rag_service.get_supabase_client'):
        service = RAGService()
        result = service.classify_query_intent(query)
        assert result['intent'] == expected_intent
        assert result['language'] == expected_language


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])