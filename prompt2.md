### Objective
Replace the hardcoded sequence list with intelligent pattern recognition that can identify when users are asking for sequence implementations versus other query types.

### Implementation Plan

#### 1. Update Search Classification in RAG Service

**File:** `pulsepal/rag_service.py`

Replace the current `classify_search_strategy` method with pattern-based detection:

```python
def classify_search_strategy(self, query: str) -> tuple[str, dict]:
    """
    Classify query using pattern recognition instead of hardcoded lists.
    
    Returns:
        tuple: (strategy_type, metadata)
        strategy_type: "vector_enhanced", "hybrid_filtered", "hybrid_full"
    """
    query_lower = query.lower()
    
    # Pattern-based sequence detection
    if self._is_sequence_query(query):
        sequence_hint = self._extract_sequence_identifier(query)
        return "vector_enhanced", {"sequence_type": sequence_hint}
    
    # API functions - use filtered hybrid
    if self._is_api_function_query(query):
        # Remove common words that pollute results
        common_words = ['function', 'method', 'example', 'sequence', 'pulse']
        filtered_words = [w for w in query.split() if w.lower() not in common_words]
        return "hybrid_filtered", {"filtered_query": ' '.join(filtered_words)}
    
    # File searches - use hybrid with URL focus
    if self._is_file_search_query(query):
        return "hybrid_full", {"focus_urls": True}
    
    # General conceptual questions - use vector search
    if self._is_conceptual_query(query):
        return "vector_enhanced", {"search_type": "documentation"}
    
    # Default to filtered hybrid
    return "hybrid_filtered", {}

def _is_sequence_query(self, query: str) -> bool:
    """
    Detect if query is asking for a sequence implementation using patterns.
    """
    query_lower = query.lower()
    
    # Direct indicators that user wants a sequence
    sequence_request_patterns = [
        r'\b(show|give|provide|need|want|have|find|create)\b.*\b(sequence|implementation|script|code|example)\b',
        r'\b(sequence|implementation|script|example|demo)\b.*\b(for|of|in)\b',
        r'\bdo you have.*sequence\b',
        r'\bany.*sequences?\b',
        r'\bhow to (write|implement|create).*sequence\b'
    ]
    
    # Check for direct sequence requests
    import re
    for pattern in sequence_request_patterns:
        if re.search(pattern, query_lower):
            return True
    
    # MRI sequence naming patterns
    mri_sequence_patterns = [
        # Uppercase abbreviations (2-6 letters) followed by sequence-related words
        r'\b[A-Z]{2,6}\b.{0,20}(sequence|implementation|script|example)',
        # write* patterns (writeHASTE, writeEPI, etc.)
        r'\bwrite[A-Z][a-zA-Z]+\b',
        # Common MRI sequence name patterns
        r'\b(spin echo|gradient echo|echo planar|turbo spin|fast spin)\b',
        # Sequences ending with common suffixes
        r'\b\w+(echo|flash|fisp|ssfp|epi|tse|fse|rare|haste|space)\b.{0,20}(sequence|script)',
        # Asking for specific sequence files
        r'\b[A-Z]{2,6}\.(m|py)\b'
    ]
    
    for pattern in mri_sequence_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return True
    
    # Check if query contains MRI sequence-like abbreviations
    # But exclude if it's clearly asking for a function (mr.makeX)
    if not re.search(r'\bmr\.\w+', query_lower):
        # Look for standalone uppercase abbreviations
        uppercase_words = re.findall(r'\b[A-Z]{2,6}\b', query)
        if uppercase_words and any(word in query_lower for word in ['sequence', 'script', 'example', 'implementation']):
            return True
    
    return False

def _is_api_function_query(self, query: str) -> bool:
    """Detect if query is asking about API functions."""
    query_lower = query.lower()
    
    # API function patterns
    api_patterns = [
        r'\bmr\.\w+',  # mr.makeBlockPulse, mr.calcDuration
        r'\bseq\.\w+',  # seq.addBlock
        r'\b(make|calc|add|get|set|check)\w+\b',  # makeTrapezoid, calcGradient
        r'\b(function|method|parameter|argument|signature)\b.*\b(for|of)\b',
        r'what (does|is)|how to use.*function'
    ]
    
    import re
    return any(re.search(pattern, query_lower) for pattern in api_patterns)

def _is_file_search_query(self, query: str) -> bool:
    """Detect if query is looking for specific files."""
    query_lower = query.lower()
    
    file_patterns = [
        r'\.(m|py|mat|seq)\b',  # File extensions
        r'\b(file|script|code)\s+(named|called)\b',
        r'\bwrite[A-Z]\w+\.(m|py)\b',  # writeHASTE.m, writeEPI.py
        r'\b(find|locate|where is)\b.*\bfile\b'
    ]
    
    import re
    return any(re.search(pattern, query_lower) for pattern in file_patterns)

def _is_conceptual_query(self, query: str) -> bool:
    """Detect if query is asking for conceptual understanding."""
    query_lower = query.lower()
    
    conceptual_patterns = [
        r'\b(what|why|how|when|explain)\b.{0,10}\b(is|are|does|do)\b',
        r'\b(concept|theory|principle|physics|understanding)\b',
        r'\b(difference between|compare|versus)\b',
        r'\btell me about\b',
        r'\bexplain\b'
    ]
    
    import re
    return any(re.search(pattern, query_lower) for pattern in conceptual_patterns)

def _extract_sequence_identifier(self, query: str) -> str:
    """
    Extract the likely sequence name from the query for better search.
    """
    import re
    
    # Try to extract sequence name patterns
    patterns_to_extract = [
        # writeXXX pattern
        (r'\bwrite([A-Z][a-zA-Z]+)\b', lambda m: m.group(1).lower()),
        # Uppercase abbreviations
        (r'\b([A-Z]{2,6})\b(?=.*sequence)', lambda m: m.group(1).lower()),
        # Common sequence names
        (r'\b(spin echo|gradient echo|echo planar|turbo spin|fast spin)\b', 
         lambda m: m.group(1).replace(' ', '_').lower()),
        # Sequences with known suffixes
        (r'\b(\w+(?:echo|flash|fisp|ssfp|epi|tse|fse|rare|haste|space))\b', 
         lambda m: m.group(1).lower())
    ]
    
    for pattern, extractor in patterns_to_extract:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return extractor(match)
    
    # Fallback: return generic "sequence"
    return "sequence"
```

#### 2. Update Score-Based Filtering

Add scoring logic that works with any sequence name:

```python
def _score_sequence_relevance(self, result: Dict, sequence_hint: str) -> float:
    """
    Score result relevance for any sequence type using patterns.
    """
    score = 0.0
    
    url = result.get("url", "").lower()
    summary = result.get("summary", "").lower()
    content = result.get("content", "").lower()
    metadata = result.get("metadata", {})
    
    # High score for write* pattern in URL
    if f"write{sequence_hint}" in url:
        score += 10.0
    elif "write" in url and sequence_hint in url:
        score += 8.0
    
    # Score for sequence name in URL (any position)
    if sequence_hint in url:
        score += 5.0
    
    # Score for sequence name in summary
    if sequence_hint in summary:
        score += 5.0
        # Extra points if it's in the first 100 characters
        if sequence_hint in summary[:100]:
            score += 2.0
    
    # Check for "implementation" or "example" near sequence name
    import re
    if re.search(f'{sequence_hint}.{{0,50}}(implementation|example|sequence)', summary + ' ' + content[:200]):
        score += 3.0
    
    # Score based on file type and language
    if metadata.get("language") in ["matlab", "python"]:
        score += 1.0
    
    # Penalize if it's clearly not a sequence
    noise_indicators = ['musical', 'melody', 'song', 'tune', 'rhythm']
    if any(noise in summary or noise in content[:500] for noise in noise_indicators):
        score = 0.0  # Zero out score for non-MRI content
    
    return score
```

#### 3. Testing Framework

Create tests to verify pattern recognition works:

```python
# tests/test_pattern_recognition.py

def test_sequence_detection_patterns():
    """Test that various sequence queries are properly detected."""
    rag_service = RAGService()
    
    # Test cases that should be detected as sequences
    sequence_queries = [
        "Do you have any HASTE sequences?",
        "Show me MOLLI implementation",
        "I need a GRE sequence example",
        "writeTSE.m",
        "How to implement FLASH sequence",
        "spin echo sequence in MATLAB",
        "Any turbo spin echo examples?",
        "MPRAGE sequence script",
        "Create EPI sequence",
        "UTE implementation please"
    ]
    
    for query in sequence_queries:
        strategy, metadata = rag_service.classify_search_strategy(query)
        assert strategy == "vector_enhanced", f"Failed to detect sequence: {query}"
        assert "sequence_type" in metadata

def test_non_sequence_detection():
    """Test that non-sequence queries are not misclassified."""
    rag_service = RAGService()
    
    # Test cases that should NOT be detected as sequences
    non_sequence_queries = [
        "What is mr.makeBlockPulse?",
        "Explain k-space theory",
        "How does T1 relaxation work?",
        "Debug my code error",
        "What parameters does calcDuration take?",
        "MRI physics concepts"
    ]
    
    for query in non_sequence_queries:
        strategy, metadata = rag_service.classify_search_strategy(query)
        assert strategy != "vector_enhanced" or metadata.get("sequence_type") != "sequence", \
            f"Incorrectly detected as sequence: {query}"

def test_sequence_name_extraction():
    """Test extraction of sequence names from queries."""
    rag_service = RAGService()
    
    test_cases = [
        ("writeHASTE.m", "haste"),
        ("MOLLI sequence implementation", "molli"),
        ("Do you have any TSE sequences?", "tse"),
        ("spin echo example", "spin_echo"),
        ("Show me the FLASH script", "flash")
    ]
    
    for query, expected in test_cases:
        sequence_name = rag_service._extract_sequence_identifier(query)
        assert sequence_name == expected, \
            f"Expected {expected}, got {sequence_name} for query: {query}"
```

### Benefits of This Approach

1. **No Hardcoding**: Works with any sequence name, including future ones
2. **Intelligent Detection**: Uses linguistic patterns to identify sequence requests
3. **Flexible**: Can handle various ways users ask for sequences
4. **Maintainable**: Easy to add new patterns without changing core logic
5. **Testable**: Clear test cases to verify behavior

### Implementation Notes

- The pattern recognition focuses on HOW users ask, not WHAT sequences exist
- Handles variations: "HASTE sequence", "writeHASTE.m", "any HASTE examples?"
- Distinguishes between sequence requests and other query types
- Falls back gracefully when patterns don't match

### Next Steps

1. Implement the pattern recognition methods
2. Test with known problematic queries (HASTE, etc.)
3. Monitor which patterns match most frequently
4. Refine patterns based on real user queries
5. Eventually supplement with a complete sequence list

Remember that PyTorch is only working in the Windows environemnt. Ask me to run any tests that would require PyTorch. Do NOT change the code for the sole reason of passing your own tests.