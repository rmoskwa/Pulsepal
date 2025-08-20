"""
Test semantic intent recognition across diverse sequence types and phrasings.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import pytest
from pulsepal.main_agent import run_pulsepal_query as run_pulsepal

class TestSemanticIntentRecognition:
    """Test that Gemini correctly understands user intent semantically."""
    
    @pytest.mark.asyncio
    async def test_implementation_intent_diversity(self):
        """Test implementation intent across different sequences and phrasings."""
        
        # Diverse sequences and phrasings - NO pattern fixation
        test_queries = [
            # Direct requests - various sequences
            "Show me a spin echo sequence",
            "Display the gradient echo code",
            "Give me TSE implementation",
            "I need MPRAGE",
            "Create a diffusion sequence",
            
            # Polite variations
            "Could you please show me UTE?",
            "Would you mind creating a spiral sequence?",
            "Can you help me with a PRESS implementation?",
            
            # Casual variations  
            "yo gimme that TrueFISP",
            "Gradient echo pls",
            "need spin echo thx",
            
            # With greetings - different sequences
            "Hello, can you show me a pulseq FSE sequence?",
            "Hi! I need to see the MPRAGE code",
            "Good morning, display gradient echo please",
            
            # Implicit requests
            "Spin echo",  # Just naming it
            "UTE sequence for short T2",
            "Implementing diffusion weighting",
            
            # How-to that implies code
            "How do I implement TSE?",
            "How to code a spiral readout?",
            "How to create gradient echo?",
        ]
        
        for query in test_queries:
            session_id, response = await run_pulsepal(query)
            
            # Should contain code
            has_code = "```matlab" in response or "```python" in response
            assert has_code, \
                f"Implementation query '{query}' should return code, got: {response[:200]}"
            
            # Should NOT ask if user wants code
            assert "Would you like to see" not in response, \
                f"Implementation query '{query}' should show code immediately"
    
    @pytest.mark.asyncio
    async def test_learning_intent_diversity(self):
        """Test learning intent with various sequences."""
        
        test_queries = [
            "What is a spin echo sequence?",
            "How does gradient echo work?",
            "Explain TSE to me",
            "Tell me about MPRAGE",
            "Why use UTE for bone imaging?",
            "When should I use spiral trajectories?",
            "Difference between TSE and FSE?",
            "Theory behind diffusion weighting?",
        ]
        
        for query in test_queries:
            session_id, response = await run_pulsepal(query)
            
            # Should NOT start with code
            first_500_chars = response[:500]
            assert "```" not in first_500_chars, \
                f"Learning query '{query}' should explain first, not show code immediately"
            
            # Should offer implementation
            assert any(phrase in response.lower() for phrase in 
                      ["would you like", "implementation", "show you the code", "see the code"]), \
                f"Learning query '{query}' should offer code after explanation"
    
    @pytest.mark.asyncio
    async def test_debug_intent_various_sequences(self):
        """Test debug intent across different sequence types."""
        
        test_queries = [
            "Maximum gradient exceeded in my TSE",
            "Spin echo timing error",
            "My gradient echo images are dark",
            "UTE sequence crashes",
            "MPRAGE contrast is wrong",
            "Spiral reconstruction artifacts",
            "undefined function mr.makeGaussPulse",
            "seq.write() not working",
        ]
        
        for query in test_queries:
            session_id, response = await run_pulsepal(query)
            
            # Should provide debugging help
            has_debug_terms = any(term in response.lower() for term in [
                'check', 'verify', 'ensure', 'try', 'solution', 
                'problem', 'fix', 'debug', 'error'
            ])
            assert has_debug_terms, \
                f"Debug query '{query}' should provide troubleshooting guidance"
    
    @pytest.mark.asyncio
    async def test_api_intent_various_functions(self):
        """Test API intent with different function namespaces."""
        
        test_queries = [
            "What parameters does mr.makeTrapezoid take?",
            "seq.addBlock syntax?",
            "How to use tra.spiral2D?",
            "mr.makeGaussPulse arguments?",
            "seq.write() parameters?",
            "Show me mr.calcDuration signature",
            "Usage of seq.setDefinition?",
        ]
        
        for query in test_queries:
            session_id, response = await run_pulsepal(query)
            
            # Should show function documentation
            has_api_terms = any(term in response.lower() for term in [
                'signature', 'parameter', 'argument', 'syntax',
                'usage', 'function', 'returns'
            ])
            assert has_api_terms, \
                f"API query '{query}' should show function documentation"
    
    @pytest.mark.asyncio
    async def test_semantic_understanding(self):
        """Test that Gemini understands meaning beyond exact keywords."""
        
        # These should all be recognized as implementation intent
        # despite very different phrasing
        implementation_variations = [
            "I require the gradient echo implementation",  # Formal
            "need gr echo",  # Abbreviated  
            "gradient echo sequence construction",  # Technical
            "can haz spin echo plz",  # Internet speak
            "Gradient echo.",  # Minimal
            "Let's build a TSE",  # Collaborative
            "Time to create MPRAGE",  # Informal
        ]
        
        for query in implementation_variations:
            session_id, response = await run_pulsepal(query)
            
            # All should be understood as wanting code
            has_code = "```" in response or "Would you like to see" in response
            assert has_code, \
                f"Semantic variation '{query}' not understood as implementation intent"
    
    @pytest.mark.asyncio  
    async def test_typos_and_mistakes(self):
        """Test robustness to typos and spelling errors."""
        
        typo_queries = [
            "Show me gradent eco sequence",  # Typos
            "I need spin ehco",  # Misspelling
            "MPRAGE implementaion",  # Common typo
            "crate a TSE sequence",  # Missing letter
            "Sprial readout code",  # Letter swap
        ]
        
        for query in typo_queries:
            session_id, response = await run_pulsepal(query)
            
            # Should still understand intent despite typos
            assert len(response) > 100, \
                f"Failed to handle typo query: '{query}'"
    
    @pytest.mark.asyncio
    async def test_context_awareness(self):
        """Test that intent recognition uses conversation context."""
        
        # First ask about concept
        session_id, response1 = await run_pulsepal("What is a gradient echo?")
        
        # Then ask for code using context
        session_id, response2 = await run_pulsepal("Now show me the code", session_id)
        
        # Should understand "the code" refers to gradient echo
        assert "```" in response2, \
            "Should understand contextual request for code"
        
        # The code should be gradient echo related
        assert any(term in response2.lower() for term in ['gradient', 'echo', 'gre', 'flash']), \
            "Code should be related to the previous gradient echo discussion"


def run_comprehensive_tests():
    """Run all semantic intent recognition tests."""
    asyncio.run(test_all())

async def test_all():
    """Execute all test suites."""
    tester = TestSemanticIntentRecognition()
    
    print("Testing Implementation Intent Diversity...")
    await tester.test_implementation_intent_diversity()
    print("✅ Implementation Intent tests passed")
    
    print("\nTesting Learning Intent...")
    await tester.test_learning_intent_diversity()
    print("✅ Learning Intent tests passed")
    
    print("\nTesting Debug Intent...")
    await tester.test_debug_intent_various_sequences()
    print("✅ Debug Intent tests passed")
    
    print("\nTesting API Intent...")
    await tester.test_api_intent_various_functions()
    print("✅ API Intent tests passed")
    
    print("\nTesting Semantic Understanding...")
    await tester.test_semantic_understanding()
    print("✅ Semantic understanding tests passed")
    
    print("\nTesting Typo Robustness...")
    await tester.test_typos_and_mistakes()
    print("✅ Typo handling tests passed")
    
    print("\nTesting Context Awareness...")
    await tester.test_context_awareness()
    print("✅ Context awareness tests passed")
    
    print("\n🎉 All semantic intent recognition tests passed!")
    print("Gemini successfully understands user intent semantically!")

if __name__ == "__main__":
    run_comprehensive_tests()