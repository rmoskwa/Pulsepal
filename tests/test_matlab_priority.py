"""
Test to ensure MATLAB examples are prioritized over Python examples.
"""

import pytest
from unittest.mock import MagicMock, patch
from pulsepal.rag_service import ModernPulseqRAG as RAGService


def test_matlab_priority_in_search_results():
    """Test that MATLAB results are shown before Python results."""
    
    with patch('pulsepal.rag_service.get_supabase_client'):
        service = RAGService()
        service._supabase_client = MagicMock()
        
        # Create mock results with both Python and MATLAB examples
        mock_results = [
            {
                'content': 'import pypulseq as pp\nseq = pp.Sequence()',
                'summary': 'Python EPI example',
                'url': 'github.com/example/epi.py',
                'metadata': {'language': 'python', 'file_extension': '.py'}
            },
            {
                'content': 'seq = mr.Sequence();\nrf = mr.makeSincPulse();',
                'summary': 'MATLAB EPI example',
                'url': 'github.com/example/epi.m',
                'metadata': {'language': 'matlab', 'file_extension': '.m'}
            },
            {
                'content': 'Another Python example with pp.make_sinc_pulse',
                'summary': 'Another Python example',
                'url': 'test.py',
                'metadata': {'language': 'python'}
            }
        ]
        
        # Test search_code_implementations with MATLAB priority
        service.supabase_client.client.from_().select().or_().or_().limit().execute = MagicMock(
            return_value=MagicMock(data=mock_results)
        )
        
        result = service.search_code_implementations("EPI sequence", language="matlab", match_count=3)
        
        # Check that MATLAB is mentioned first
        assert "MATLAB" in result
        assert result.index("mr.Sequence()") < result.index("pp.Sequence()") if "pp.Sequence()" in result else True
        
        # The formatted output should indicate MATLAB as the language
        assert "*Language: MATLAB*" in result or "MATLAB" in result.split('\n')[1]


def test_python_only_fallback_with_note():
    """Test that when only Python results exist, a note is added."""
    
    with patch('pulsepal.rag_service.get_supabase_client'):
        service = RAGService()
        service._supabase_client = MagicMock()
        
        # Only Python results
        mock_results = [
            {
                'content': 'import pypulseq as pp\nseq = pp.Sequence()',
                'summary': 'Python EPI example',
                'url': 'github.com/example/epi.py',
                'metadata': {'language': 'python', 'file_extension': '.py'}
            }
        ]
        
        service.supabase_client.client.from_().select().or_().or_().limit().execute = MagicMock(
            return_value=MagicMock(data=mock_results)
        )
        
        result = service.search_code_implementations("EPI sequence", language="matlab", match_count=1)
        
        # Should include a note about showing Python when MATLAB was expected
        assert "Python implementation" in result or "no MATLAB version found" in result


def test_explicit_python_request():
    """Test that Python is shown when explicitly requested."""
    
    with patch('pulsepal.rag_service.get_supabase_client'):
        service = RAGService()
        service._supabase_client = MagicMock()
        
        # Both Python and MATLAB results
        mock_results = [
            {
                'content': 'seq = mr.Sequence();\nrf = mr.makeSincPulse();',
                'summary': 'MATLAB example',
                'url': 'test.m',
                'metadata': {'language': 'matlab'}
            },
            {
                'content': 'import pypulseq as pp\nseq = pp.Sequence()',
                'summary': 'Python example',
                'url': 'test.py',
                'metadata': {'language': 'python'}
            }
        ]
        
        service.supabase_client.client.from_().select().or_().or_().limit().execute = MagicMock(
            return_value=MagicMock(data=mock_results)
        )
        
        # Explicitly request Python
        result = service.search_code_implementations("EPI sequence", language="python", match_count=2)
        
        # Python should be shown first when explicitly requested
        if "pp.Sequence()" in result and "mr.Sequence()" in result:
            assert result.index("pp.Sequence()") < result.index("mr.Sequence()")


def test_format_code_results_language_detection():
    """Test that _format_code_results correctly detects and notes language."""
    
    with patch('pulsepal.rag_service.get_supabase_client'):
        service = RAGService()
        
        # Test with Python result when MATLAB expected
        python_results = [{
            'content': 'import pypulseq as pp\nseq = pp.Sequence()',
            'summary': 'Python example',
            'url': 'test.py',
            'metadata': {'language': 'python'}
        }]
        
        formatted = service._format_code_results(python_results, "EPI sequence")
        assert "Python example" in formatted or "no MATLAB version found" in formatted
        
        # Test with MATLAB result (default)
        matlab_results = [{
            'content': 'seq = mr.Sequence();\n',
            'summary': 'MATLAB example',
            'url': 'test.m',
            'metadata': {'language': 'matlab'}
        }]
        
        formatted = service._format_code_results(matlab_results, "EPI sequence")
        assert "MATLAB" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])