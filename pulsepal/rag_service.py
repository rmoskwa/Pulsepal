"""
RAG (Retrieval Augmented Generation) service for Pulsepal.

This module provides the core RAG functionality for searching documentation
and code examples, with support for various search modes and result formatting.
"""

import logging
from typing import List, Dict, Any, Optional
from enum import Enum
from .supabase_client import get_supabase_client, SupabaseRAGClient
from .settings import get_settings
from .rag_performance import get_performance_monitor

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """Types of searches based on query classification."""

    API_FUNCTION = "api_function"
    CODE_EXAMPLE = "code_example"
    DOCUMENTATION = "documentation"
    UNIFIED = "unified"


class RAGService:
    """Service for performing RAG queries on Pulseq documentation and code."""

    def __init__(self):
        """Initialize RAG service with Supabase client."""
        self._supabase_client = None
        self.settings = get_settings()
        self.performance_monitor = get_performance_monitor()

        # Configurable content preview limits - increased to provide full context to AI
        self.doc_preview_limit = 5000  # Was hardcoded 500
        self.code_preview_limit = 1500  # Reduced from 10000 to prevent RECITATION errors

    @property
    def supabase_client(self) -> SupabaseRAGClient:
        """Lazy load Supabase client with retry logic."""
        if self._supabase_client is None:
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    self._supabase_client = get_supabase_client()
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Failed to connect to Supabase (attempt {attempt + 1}/{max_retries}): {e}")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"Failed to connect to Supabase after {max_retries} attempts")
                        raise ConnectionError(f"Unable to connect to knowledge base after {max_retries} attempts") from e
        
        return self._supabase_client

    def classify_search_strategy(self, query: str) -> tuple[str, dict]:
        """
        Classify query using pattern recognition instead of hardcoded lists.
        
        Returns:
            tuple: (strategy_type, metadata)
            strategy_type: "vector_enhanced", "hybrid_filtered", "hybrid_full"
        """
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

    def classify_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Classify user query to determine optimal search strategy.
        
        Intent categories:
        - 'example_request': User wants code implementation
          Keywords: "example", "implement", "create", "write", "build"
          Action: Search .m/.py files in crawled_pages
          
        - 'function_lookup': User asking about specific function
          Keywords: "mr.", "seq.", "parameters", "syntax", "how to use"
          Action: Search function_calling_patterns view
          
        - 'debug_request': User has code problems/errors  
          Keywords: "error", "bug", "wrong", "debug", "not working", "dark image"
          Action: Validate functions + search for solutions
          
        - 'tutorial_request': User wants step-by-step learning
          Keywords: "tutorial", "learn", "new to", "explain step by step"
          Action: Search notebooks/PDFs with educational structure
          
        - 'concept_question': User asking about Pulseq concepts
          Keywords: "how do I", "what is", "why"
          Action: Search documentation in crawled_pages
        
        Returns dict with:
        - 'intent': Primary intent category
        - 'confidence': Float 0-1
        - 'language': 'matlab' or 'python' if specified
        - 'search_strategy': Specific search approach
        """
        query_lower = query.lower()

        # Intent patterns with keywords and detection logic
        example_keywords = [
            "example", "implement", "create", "write", "build", "show me", 
            "code", "script", "demo", "sample", "how to make", "sequence for"
        ]
        
        function_keywords = [
            "mr.", "seq.", "parameters", "syntax", "how to use", "function",
            "method", "api", "signature", "returns", "arguments", "usage",
            "what does", "calling pattern"
        ]
        
        debug_keywords = [
            "error", "bug", "wrong", "debug", "not working", "dark image",
            "undefined function", "maximum gradient exceeded", "issue", "problem",
            "fails", "crash", "doesn't work", "incorrect"
        ]
        
        tutorial_keywords = [
            "tutorial", "learn", "new to", "explain step by step", "guide",
            "walkthrough", "teach", "beginner", "introduction", "getting started",
            "step-by-step", "lesson"
        ]
        
        concept_keywords = [
            "what is", "how does", "why", "when", "theory", "concept",
            "physics", "principle", "understand", "explain", "overview",
            "difference between", "compare"
        ]

        # Known Pulseq functions (common ones)
        pulseq_functions = [
            "makearbitraryrf", "maketrapezoid", "makegausspulse", "makeblockpulse",
            "makesincpulse", "makeadc", "makedelay", "makelabel", "maketrigger",
            "calcduration", "calcrfbandwidth", "calcrfphase", "writeute", "writeflash",
            "writeepi", "writespinecho", "writetse", "writegradientecho", "writebssfp",
            "addblock", "write", "plot", "setdefinition"
        ]

        # Score each intent category
        scores = {
            'example_request': sum(1 for kw in example_keywords if kw in query_lower),
            'function_lookup': sum(1 for kw in function_keywords if kw in query_lower) + 
                              sum(2 for func in pulseq_functions if func in query_lower),
            'debug_request': sum(2 for kw in debug_keywords if kw in query_lower),  # Higher weight for debug
            'tutorial_request': sum(1.5 for kw in tutorial_keywords if kw in query_lower),
            'concept_question': sum(1 for kw in concept_keywords if kw in query_lower)
        }

        # Detect language preference (default to MATLAB)
        language = "matlab"  # Default
        if "python" in query_lower or ".py" in query_lower or "pypulseq" in query_lower:
            language = "python"
        elif "matlab" in query_lower or ".m" in query_lower:
            language = "matlab"  # Explicit MATLAB
        elif "c++" in query_lower or "cpp" in query_lower:
            language = "cpp"

        # Determine primary intent based on highest score
        max_score = max(scores.values())
        if max_score == 0:
            # No clear intent - use unified search
            primary_intent = "concept_question"  # Default to concept
            confidence = 0.3
            search_strategy = "unified"
        else:
            # Get the intent with highest score
            primary_intent = max(scores, key=scores.get)
            
            # Calculate confidence based on score strength
            total_indicators = sum(1 for s in scores.values() if s > 0)
            if total_indicators == 1 and max_score >= 2:
                confidence = 0.9  # Very clear intent
            elif total_indicators == 1:
                confidence = 0.7  # Clear intent
            elif max_score >= 3:
                confidence = 0.8  # Strong signal despite mixed intents
            else:
                confidence = 0.5  # Mixed signals
            
            # Determine search strategy based on intent
            strategy_map = {
                'example_request': 'code_search',
                'function_lookup': 'api_enhanced',
                'debug_request': 'debug_validate',
                'tutorial_request': 'tutorial_search',
                'concept_question': 'documentation'
            }
            search_strategy = strategy_map.get(primary_intent, 'unified')

        # Check for specific patterns that override classification
        if "mr.write(" in query_lower or "undefined function 'mr.write'" in query_lower:
            primary_intent = "debug_request"
            confidence = 1.0
            search_strategy = "debug_validate"
        elif self._is_sequence_query(query):
            primary_intent = "example_request"
            confidence = 0.9
            search_strategy = "code_search"

        return {
            "intent": primary_intent,
            "confidence": confidence,
            "language": language,
            "search_strategy": search_strategy,
            "scores": scores,  # Include raw scores for transparency
            # Backward compatibility
            "primary_type": SearchType.CODE_EXAMPLE if primary_intent == "example_request" else
                           SearchType.API_FUNCTION if primary_intent == "function_lookup" else
                           SearchType.DOCUMENTATION,
            "search_api": primary_intent == "function_lookup",
            "search_code": primary_intent in ["example_request", "tutorial_request"],
            "search_docs": primary_intent in ["concept_question", "tutorial_request"]
        }

    def _create_search_variations(self, query: str) -> List[str]:
        """
        Generate search variations for more forgiving search.

        Args:
            query: Original search query

        Returns:
            List of query variations to try
        """
        import re

        variations = []

        # Original case
        variations.append(query)

        # Case variations
        variations.extend([query.lower(), query.upper(), query.capitalize()])

        # Handle snake_case vs camelCase
        # Convert snake_case to camelCase
        if "_" in query:
            parts = query.split("_")
            camel = parts[0].lower() + "".join(p.capitalize() for p in parts[1:])
            variations.append(camel)

        # Convert camelCase to snake_case (basic)
        snake = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", query)
        snake = re.sub("([a-z0-9])([A-Z])", r"\1_\2", snake).lower()
        if snake != query.lower():
            variations.append(snake)

        # Common Pulseq variations
        query_lower = query.lower()

        # Handle 'make' prefix variations
        if query_lower.startswith("make"):
            variations.append("mr." + query)
            variations.append("mr." + query_lower)

        # Handle 'mr.' prefix variations
        if query_lower.startswith("mr."):
            variations.append(query[3:])
            variations.append(query_lower[3:])

        # Handle potential class methods (remove 'seq.' if present)
        if query_lower.startswith("seq."):
            variations.append(query[4:])
            variations.append(query_lower[4:])

        # Add potential class method notation
        if "." not in query_lower and not query_lower.startswith("make"):
            variations.append("seq." + query)
            variations.append("seq." + query_lower)

        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for v in variations:
            if v and v not in seen:
                seen.add(v)
                unique_variations.append(v)

        logger.debug(f"Generated variations for '{query}': {unique_variations[:5]}")
        return unique_variations
    
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
            # Uppercase abbreviations (2-6 letters) optionally followed by sequence-related words
            r'\b[A-Z]{2,6}\b(?:$|\s|\.|\s+(?:sequence|implementation|script|example))',
            # write* patterns (writeHASTE, writeEPI, etc.)
            r'\bwrite[A-Z][a-zA-Z]+\b',
            # Common MRI sequence name patterns
            r'\b(spin echo|gradient echo|echo planar|turbo spin|fast spin)\b',
            # Sequences ending with common suffixes
            r'\b\w+(echo|flash|fisp|ssfp|epi|tse|fse|rare|haste|space)\b',
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
        
        # Try to extract sequence name patterns (order matters!)
        patterns_to_extract = [
            # writeXXX pattern (highest priority)
            (r'\bwrite([A-Z][a-zA-Z]+)\b', lambda m: m.group(1).lower()),
            # Common sequence names (before uppercase check)
            (r'\b(spin echo|gradient echo|echo planar|turbo spin|fast spin)\b', 
             lambda m: m.group(1).replace(' ', '_').lower()),
            # Uppercase abbreviations that are likely sequence names (2-6 letters)
            # Look for sequences like TSE, EPI, FLASH, etc.
            (r'\b([A-Z]{2,6})\b(?:\s+(?:sequence|implementation|script|example|sequences)|$)', 
             lambda m: m.group(1).lower()),
            # Standalone uppercase words at end of sentence or followed by punctuation
            (r'\b([A-Z]{2,6})\b(?:$|[.,!?]|\s*$)', 
             lambda m: m.group(1).lower()),
            # Sequences with known suffixes (but not common words)
            (r'\b(?!can|you|show|give|need|want|have|find|create)(\w+(?:echo|flash|fisp|ssfp|epi|tse|fse|rare|haste|space))\b', 
             lambda m: m.group(1).lower() if len(m.group(1)) > 2 else None)
        ]
        
        for pattern, extractor in patterns_to_extract:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                result = extractor(match)
                if result:  # Skip None results
                    return result
        
        # Fallback: return generic "sequence"
        return "sequence"
    
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
        
        # Score based on file type and language - MATLAB gets higher priority
        language = metadata.get("language", "").lower()
        if language == "matlab":
            score += 5.0  # Strong preference for MATLAB
        elif language == "python":
            score += 1.0  # Still add points for Python, but less than MATLAB
        
        # Penalize if it's clearly not a sequence
        noise_indicators = ['musical', 'melody', 'song', 'tune', 'rhythm']
        if any(noise in summary or noise in content[:500] for noise in noise_indicators):
            score = 0.0  # Zero out score for non-MRI content
        
        return score

    async def search_functions_fast(self, query: str, limit: int = 10) -> list[dict]:
        """
        Phase 1: Lightweight function discovery using api_reference table.
        Returns only essential fields for <50ms response time.
        """
        try:
            # Try RPC first if it exists
            try:
                result = await self.supabase_client.rpc(
                    'match_api_reference_search',
                    {
                        'query_text': query,
                        'match_count': limit
                    }
                ).execute()
                
                if result.data:
                    return result.data
            except:
                pass  # RPC doesn't exist, use fallback
            
            # Fallback to direct table search
            result = await self.supabase_client.table('api_reference')\
                .select('name, signature, description, calling_pattern, is_class_method')\
                .ilike('name', f'%{query}%')\
                .limit(limit)\
                .execute()
                
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Fast function search failed: {e}")
            return []
    
    async def get_function_details(self, function_names: list[str]) -> list[dict]:
        """
        Phase 2: Get complete function details using api_reference table.
        Only called when generating actual code.
        """
        try:
            # Use the api_reference table directly since api_reference_details view may not exist
            result = await self.supabase_client.table('api_reference')\
                .select('name, signature, parameters, usage_examples, returns, has_nargin_pattern, calling_pattern')\
                .in_('name', function_names)\
                .execute()
                
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Failed to get function details for {function_names}: {e}")
            return []
    
    async def get_official_sequence(self, sequence_type: str) -> dict:
        """
        Get official validated sequence from official_sequence_examples table.
        Optimized to use ai_summary for fast retrieval, only fetching content when needed.
        """
        try:
            # Normalize sequence type (handle variations)
            type_mapping = {
                'epi': 'EPI',
                'echo_planar': 'EPI',
                'spin_echo': 'SpinEcho',
                'spinecho': 'SpinEcho',
                'gradient_echo': 'GradientEcho',
                'gre': 'GradientEcho',
                'tse': 'TSE',
                'turbo_spin': 'TSE',
                'mprage': 'MPRAGE',
                'ute': 'UTE',
                'haste': 'HASTE',
                'trufisp': 'TrueFISP',
                'press': 'PRESS',
                'spiral': 'Spiral'
            }
            
            normalized_type = type_mapping.get(sequence_type.lower(), sequence_type)
            
            # First try official_sequence_examples with optimized two-step query
            try:
                # Step 1: Fast query using ai_summary (only ~2KB per record)
                # Get file_name and ai_summary to identify the best match
                # Note: Using file_name as unique identifier since table has no id column
                summary_result = self.supabase_client.client.from_('official_sequence_examples')\
                    .select('file_name, sequence_type, ai_summary')\
                    .eq('sequence_type', normalized_type)\
                    .order('file_name')\
                    .limit(1)\
                    .execute()
                    
                if summary_result.data and len(summary_result.data) > 0:
                    # Extract record info from step 1
                    record = summary_result.data[0]
                    file_name = record['file_name']
                    
                    logger.debug(f"Found sequence in {file_name}, fetching content...")
                    
                    # Step 2: Fetch ONLY the content field for this specific record
                    # Using file_name as the unique identifier
                    content_result = self.supabase_client.client.from_('official_sequence_examples')\
                        .select('content')\
                        .eq('file_name', file_name)\
                        .limit(1)\
                        .execute()
                    
                    if content_result.data and len(content_result.data) > 0:
                        content = content_result.data[0].get('content', '')
                        if content:
                            logger.debug(f"Got content, length: {len(content)} chars")
                            return {
                                'content': content,
                                'file_name': file_name,
                                'sequence_type': record['sequence_type'],
                                'ai_summary': record.get('ai_summary', '')  # Include summary for reference
                            }
                        else:
                            logger.warning(f"Content field is empty for id={record_id}")
                    else:
                        logger.warning(f"No content record found for id={record_id}")
            except Exception as e:
                logger.debug(f"official_sequence_examples query failed: {e}")
                pass  # Table might not exist, fallback to search
            
            # Fallback: Search in crawled_pages for official sequences
            # Use a more targeted search to avoid timeouts
            search_query = f"{normalized_type} sequence"
            
            # Use the supabase client's search with limited fields
            try:
                # Search using vector similarity on summaries
                raw_results = self.supabase_client.perform_hybrid_search(
                    query=search_query,
                    match_count=5,  # Get a few candidates
                    search_type="code_examples"
                )
                
                if raw_results:
                    # Look for official sequences in results
                    for result in raw_results:
                        url = result.get('url', '').lower()
                        content = result.get('content', '')
                        
                        # Check if it's from official demoSeq
                        if 'demoseq' in url or 'official' in url:
                            # Parse content to get just the code part
                            if '---' in content:
                                # Split on --- to get the full content after summary
                                parts = content.split('---', 1)
                                if len(parts) > 1:
                                    content = parts[1].strip()
                            
                            return {
                                'content': content,
                                'file_name': f'{normalized_type}_example.m',
                                'sequence_type': normalized_type
                            }
                    
                    # If no official found, return the best match
                    best_match = raw_results[0]
                    content = best_match.get('content', '')
                    
                    # Parse content to get full code after summary
                    if '---' in content:
                        parts = content.split('---', 1)
                        if len(parts) > 1:
                            content = parts[1].strip()
                    
                    return {
                        'content': content,
                        'file_name': f'{normalized_type}_example.m',
                        'sequence_type': normalized_type
                    }
                    
            except Exception as e:
                logger.error(f"Fallback search failed: {e}")
                
            return None
            
        except Exception as e:
            logger.error(f"Official sequence fetch failed: {e}")
            return None

    async def search_api_functions_enhanced(self, query: str, language: str = "matlab", match_count: int = 5) -> str:
        """
        Search for Pulseq functions using the function_calling_patterns view.
        
        This method:
        1. Queries the function_calling_patterns view for exact calling patterns
        2. Prioritizes exact function name matches
        3. Returns formatted results with:
           - Correct calling pattern (e.g., mr.makeTrapezoid or seq.write)
           - Usage instructions (for class methods that need instantiation)
           - Parameters and description
        4. Handles both regular functions and class methods appropriately
        
        Returns formatted string with clear usage examples.
        """
        context = self.performance_monitor.start_query(query, "api_functions_enhanced")
        context["language"] = language
        
        try:
            # Try to use the function_calling_patterns view first
            try:
                # Direct query to the function_calling_patterns view
                query_builder = self.supabase_client.client.from_("function_calling_patterns").select("*")
                
                # Search across multiple fields
                search_terms = query.lower().replace("mr.", "").replace("seq.", "")
                query_builder = query_builder.or_(
                    f"function_name.ilike.%{search_terms}%,"
                    f"description.ilike.%{search_terms}%,"
                    f"calling_pattern.ilike.%{search_terms}%"
                )
                
                # Language filter - default to MATLAB
                if language.lower() == "matlab":
                    query_builder = query_builder.eq("language", "matlab")
                elif language.lower() == "python":
                    query_builder = query_builder.eq("language", "python")
                
                # Execute query
                result = query_builder.limit(match_count).execute()
                enhanced_results = result.data if result.data else []
                
                # If no results from view, fall back to regular API search
                if not enhanced_results:
                    logger.info("No results from function_calling_patterns view, falling back to regular search")
                    return self.search_api_functions(query, match_count, language)
                
                # Format enhanced results
                formatted = [f"## Pulseq Function Usage for: '{query}'\n"]
                formatted.append(f"*Language: {language.upper()}*\n")
                formatted.append(f"Found {len(enhanced_results)} function(s):\n")
                
                for i, func in enumerate(enhanced_results, 1):
                    function_name = func.get("function_name", "Unknown")
                    calling_pattern = func.get("calling_pattern", "")
                    usage_instruction = func.get("usage_instruction", "")
                    parameters = func.get("parameters", "")
                    description = func.get("description", "")
                    is_class_method = func.get("is_class_method", False)
                    
                    formatted.append(f"### {i}. {function_name}")
                    
                    # Highlight if it's a class method
                    if is_class_method:
                        formatted.append("**Type:** Class Method (requires instance)")
                    
                    formatted.append(f"**Correct Usage:** `{calling_pattern}`")
                    
                    if usage_instruction:
                        formatted.append(f"**Instructions:** {usage_instruction}")
                    
                    if description:
                        formatted.append(f"**Description:** {description}")
                    
                    if parameters:
                        formatted.append(f"**Parameters:** {parameters}")
                    
                    # Add quick example based on pattern
                    if is_class_method:
                        if language.lower() == "matlab":
                            formatted.append("**Quick Example:**")
                            formatted.append("```matlab")
                            formatted.append("% First create the sequence object")
                            formatted.append("seq = mr.Sequence();")
                            formatted.append("% Then use the method")
                            formatted.append(f"{calling_pattern};")
                            formatted.append("```")
                        else:
                            formatted.append("**Quick Example:**")
                            formatted.append("```python")
                            formatted.append("# First create the sequence object")
                            formatted.append("from pypulseq import Sequence")
                            formatted.append("seq = Sequence()")
                            formatted.append("# Then use the method")
                            formatted.append(f"{calling_pattern}")
                            formatted.append("```")
                    
                    formatted.append("")  # Empty line between results
                
                self.performance_monitor.record_query_completion(context, enhanced_results)
                return "\n".join(formatted)
                
            except Exception as view_error:
                logger.warning(f"function_calling_patterns view query failed: {view_error}")
                # Fall back to regular search
                return self.search_api_functions(query, match_count, language)
                
        except Exception as e:
            self.performance_monitor.record_query_completion(context, [], error=str(e))
            logger.error(f"Enhanced API function search failed: {e}")
            return self.search_api_functions(query, match_count, language)

    async def search_api_functions(
        self, query: str, match_count: int = 5, language_filter: Optional[str] = None
    ) -> str:
        """
        Enhanced API function search with two-tier approach for performance.
        Phase 1: Fast search with minimal fields
        Phase 2: Get full details only for functions needed for code generation

        Args:
            query: Search query
            match_count: Number of results to return
            language_filter: Optional language filter (matlab/python/cpp)

        Returns:
            Formatted search results with transparency
        """
        context = self.performance_monitor.start_query(query, "api_functions")
        context["language_filter"] = language_filter

        try:
            # Phase 1: Fast search with minimal fields
            fast_results = await self.search_functions_fast(query, match_count * 2)
            
            # Filter by language if specified
            if language_filter:
                fast_results = [r for r in fast_results if r.get("language") == language_filter]
            elif "python" not in query.lower() and "pypulseq" not in query.lower():
                # Default to MATLAB preference
                matlab_results = [r for r in fast_results if r.get("language") == "matlab"]
                other_results = [r for r in fast_results if r.get("language") != "matlab"]
                fast_results = matlab_results + other_results
            
            # Limit to requested count
            final_results = fast_results[:match_count]
            
            # Phase 2 is only called when actually generating code
            # For now, just return the fast search results for display
            
            # Record successful completion
            self.performance_monitor.record_query_completion(context, final_results)

            # Format results with transparency about the search process
            return self._format_api_results_with_transparency(
                final_results, query, [query]
            )

        except Exception as e:
            # Record failure
            self.performance_monitor.record_query_completion(context, [], error=str(e))
            logger.error(f"API function search failed: {e}")
            return f"Error searching API functions: {str(e)}"

    def _fallback_api_search(
        self, query: str, match_count: int, language_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fallback API search using direct SQL query.
        """
        try:
            query_builder = self.supabase_client.client.from_("api_reference").select(
                "*"
            )

            # Add text search across multiple columns
            query_builder = query_builder.or_(
                f"name.ilike.%{query}%,"
                f"description.ilike.%{query}%,"
                f"signature.ilike.%{query}%"
            )

            # Add language filter if specified
            if language_filter:
                query_builder = query_builder.eq("language", language_filter)

            # Execute query
            result = query_builder.limit(match_count).execute()

            if result.data:
                logger.info(f"Fallback API search found {len(result.data)} results")
                return result.data
            else:
                return []

        except Exception as e:
            logger.error(f"Fallback API search failed: {e}")
            return []

    def search_all_sources(self, query: str, match_count: int = 10) -> str:
        """
        Search across all data sources based on query type.

        Args:
            query: Search query
            match_count: Total number of results to return

        Returns:
            Formatted search results from all relevant sources
        """
        # Classify query
        strategy = self.classify_query_intent(query)
        logger.debug(f"Query classification: {strategy}")

        results = {
            "api_functions": [],
            "code_examples": [],
            "documentation": [],
            "strategy": strategy,
        }

        # Determine result distribution based on classification
        if strategy["primary_type"] == SearchType.API_FUNCTION:
            # Prioritize API results
            api_count = max(1, match_count // 2)
            code_count = max(1, match_count // 4)
            doc_count = match_count - api_count - code_count
        elif strategy["primary_type"] == SearchType.CODE_EXAMPLE:
            # Prioritize code examples
            code_count = max(1, match_count // 2)
            api_count = max(1, match_count // 4)
            doc_count = match_count - api_count - code_count
        elif strategy["primary_type"] == SearchType.DOCUMENTATION:
            # Prioritize documentation
            doc_count = max(1, match_count // 2)
            code_count = max(1, match_count // 4)
            api_count = match_count - code_count - doc_count
        else:
            # Unified - equal distribution
            api_count = max(1, match_count // 3)
            code_count = max(1, match_count // 3)
            doc_count = match_count - api_count - code_count

        # Search each relevant source
        try:
            if strategy["search_api"] and api_count > 0:
                api_results = self.search_api_functions(
                    query, api_count, strategy.get("language")
                )
                # Extract results from formatted string (not ideal but maintains compatibility)
                if "No API functions found" not in api_results:
                    results["api_functions"] = [{"formatted_result": api_results}]

            if strategy["search_code"] and code_count > 0:
                code_results = self.search_code_examples(
                    query, match_count=code_count, use_hybrid=True
                )
                if "No code examples found" not in code_results:
                    results["code_examples"] = [{"formatted_result": code_results}]

            if strategy["search_docs"] and doc_count > 0:
                doc_results = self.perform_rag_query(
                    query, match_count=doc_count, use_hybrid=True
                )
                if "No documentation found" not in doc_results:
                    results["documentation"] = [{"formatted_result": doc_results}]

        except Exception as e:
            logger.error(f"Error in unified search: {e}")

        return self._format_unified_results(results, query)

    def _format_api_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Format API search results for display."""
        if not results:
            return f"No API functions found for query: '{query}'"

        formatted = [f"## Pulseq API Functions for: '{query}'\n"]
        formatted.append(f"Found {len(results)} function(s):\n")

        for i, func in enumerate(results, 1):
            # Extract function details
            name = func.get("name", "Unknown")
            language = func.get("language", "Unknown")
            signature = func.get("signature", "")
            description = func.get("description", "")
            parameters = func.get("parameters", {})
            returns = func.get("returns", "")
            similarity = func.get("similarity", 0)

            # Format result
            formatted.append(f"### {i}. {name} ({language.upper()})")
            formatted.append(f"**Relevance:** {similarity:.2%}")

            if signature:
                formatted.append("**Signature:**")
                formatted.append(f"```{language}")
                formatted.append(signature)
                formatted.append("```")

            if description:
                formatted.append(f"**Description:** {description}")

            if parameters and isinstance(parameters, dict):
                formatted.append("**Parameters:**")
                for param, details in parameters.items():
                    if isinstance(details, dict):
                        param_type = details.get("type", "unknown")
                        param_desc = details.get("description", "No description")
                        formatted.append(f"- `{param}` ({param_type}): {param_desc}")
                    else:
                        formatted.append(f"- `{param}`: {details}")

            if returns:
                formatted.append(f"**Returns:** {returns}")

            formatted.append("")  # Empty line between results

        return "\n".join(formatted)

    def _format_api_results_with_transparency(
        self,
        results: List[Dict[str, Any]],
        original_query: str,
        variations_tried: List[str],
    ) -> str:
        """
        Format API results with full transparency about search process.

        Args:
            results: Search results
            original_query: Original user query
            variations_tried: List of variations that were searched

        Returns:
            Formatted results with transparency notes
        """
        if not results:
            return self._format_helpful_not_found(original_query, variations_tried)

        formatted = [f"## Search Results for: '{original_query}'\n"]

        # Be transparent about search process if variations were used
        if len(variations_tried) > 1 and results:
            # Check if we found exact match or had to use variations
            result_names = [r.get("name", "").lower() for r in results]
            original_lower = original_query.lower()

            # Check if original query appears in any result
            found_exact = any(
                original_lower in name or name in original_lower
                for name in result_names
            )

            if not found_exact:
                formatted.append(
                    "*Note: Exact match not found. Showing similar functions based on variations.*\n"
                )

        # Group by language if mixed results
        languages = set(r.get("language") for r in results if r.get("language"))
        if len(languages) > 1:
            formatted.append(
                f"*Found implementations in: {', '.join(sorted(languages))}*\n"
            )
            formatted.append("*MATLAB results are shown first by default.*\n")

        formatted.append(f"Found {len(results)} function(s):\n")

        # Format each result with language clarity
        for i, func in enumerate(results, 1):
            name = func.get("name", "Unknown")
            language = func.get("language", "Unknown")
            signature = func.get("signature", "")
            description = func.get("description", "")
            parameters = func.get("parameters", {})
            returns = func.get("returns", "")
            similarity = func.get("similarity", 0)

            # Add language tag to make it clear
            formatted.append(f"### {i}. {name} ({language.upper()})")

            # If showing Python result when MATLAB was likely intended
            if (
                language == "python"
                and "matlab" not in original_query.lower()
                and "python" not in original_query.lower()
            ):
                formatted.append(
                    "*Note: This is the Python (pypulseq) version. MATLAB equivalent may have slight syntax differences.*"
                )

            formatted.append(f"**Relevance:** {similarity:.2%}")

            if signature:
                formatted.append("**Signature:**")
                formatted.append(f"```{language}")
                formatted.append(signature)
                formatted.append("```")

            if description:
                formatted.append(f"**Description:** {description}")

            if parameters and isinstance(parameters, dict):
                formatted.append("**Parameters:**")
                for param, details in parameters.items():
                    if isinstance(details, dict):
                        param_type = details.get("type", "unknown")
                        param_desc = details.get("description", "No description")
                        formatted.append(f"- `{param}` ({param_type}): {param_desc}")
                    else:
                        formatted.append(f"- `{param}`: {details}")

            if returns:
                formatted.append(f"**Returns:** {returns}")

            formatted.append("")  # Empty line between results

        return "\n".join(formatted)

    def _format_helpful_not_found(self, query: str, variations_tried: List[str]) -> str:
        """
        Provide helpful message when function not found.

        Args:
            query: Original search query
            variations_tried: List of variations that were tried

        Returns:
            Helpful not-found message with suggestions
        """
        formatted = [f"## No Results Found for: '{query}'\n"]

        # Check if it might be a class method
        query_lower = query.lower()
        if (
            "." not in query_lower
            and not query_lower.startswith("make")
            and not query_lower.startswith("mr")
        ):
            formatted.append("**This might be a method of the Sequence class.**")
            formatted.append("\nIn MATLAB, try:")
            formatted.append("```matlab")
            formatted.append("% Create sequence object first")
            formatted.append("seq = mr.opts();  % or your existing sequence object")
            formatted.append(f"seq.{query}(...);  % If it's a Sequence method")
            formatted.append("```")
            formatted.append("\nIn Python (pypulseq), try:")
            formatted.append("```python")
            formatted.append("# Create sequence object first")
            formatted.append("seq = Sequence()  # or your existing sequence object")
            formatted.append(f"seq.{query}(...)  # If it's a Sequence method")
            formatted.append("```")

        # Show what we tried (but limit to avoid clutter)
        if len(variations_tried) > 1:
            shown_variations = variations_tried[1:4]  # Show up to 3 variations
            formatted.append(
                f"\n*I also searched for variations: {', '.join(shown_variations)}*"
            )

        # Suggest alternatives
        formatted.append("\n**Would you like me to:**")
        formatted.append("1. Search for similar function names in the codebase")
        formatted.append("2. Look for code examples that might use this functionality")
        formatted.append("3. Explain the general concept if it's an MRI physics term")
        formatted.append(
            "\nPlease provide more context or try a different search term."
        )

        return "\n".join(formatted)

    def _format_unified_results(
        self, results: Dict[str, List[Dict]], query: str
    ) -> str:
        """
        Format unified search results for display.
        """
        formatted = [f"## Search Results for: '{query}'\n"]

        strategy = results.get("strategy", {})
        confidence = strategy.get("confidence", 0)

        if confidence < 0.5:
            formatted.append(
                "*Note: Query intent unclear, showing results from all sources*\n"
            )
        else:
            primary_type = strategy.get("primary_type", SearchType.UNIFIED)
            formatted.append(
                f"*Detected intent: {primary_type.value.replace('_', ' ').title()}*\n"
            )

        # Show results from each source
        sources_shown = 0

        # API Functions
        if results["api_functions"]:
            formatted.append("### 🔧 API Functions\n")
            for result in results["api_functions"]:
                formatted.append(result["formatted_result"])
            sources_shown += 1

        # Code Examples
        if results["code_examples"]:
            if sources_shown > 0:
                formatted.append("---\n")
            formatted.append("### 💻 Code Examples\n")
            for result in results["code_examples"]:
                formatted.append(result["formatted_result"])
            sources_shown += 1

        # Documentation
        if results["documentation"]:
            if sources_shown > 0:
                formatted.append("---\n")
            formatted.append("### 📚 Documentation\n")
            for result in results["documentation"]:
                formatted.append(result["formatted_result"])
            sources_shown += 1

        if sources_shown == 0:
            formatted.append("No results found in any source for this query.")

        return "\n".join(formatted)

    def format_results_adaptive(self, results: List[Dict], query_intent: str, query: str) -> str:
        """
        Format search results based on user intent.
        
        Templates by intent:
        
        'function_lookup':
            ## {function_name}
            **Calling Pattern:** `{correct_usage}`
            **Usage:** {usage_instruction}
            **Parameters:** {params}
            **Quick Example:**
            ```matlab
            {example}
            ```
        
        'example_request':
            ```matlab
            {full_code}
            ```
            **Key Points:** {brief_explanation}
            **Functions Used:** {function_list}
        
        'debug_request':
            ## Issue Found
            {error_description}
            
            **Problem:** {specific_issue}
            **Solution:** {how_to_fix}
            
            **Corrected Code:**
            ```matlab
            {fixed_code}
            ```
        
        'tutorial_request':
            ## {concept_title}
            
            ### Understanding the Concept
            {explanation}
            
            ### Step-by-Step Implementation
            {numbered_steps_with_code}
            
            ### Complete Example
            {full_annotated_code}
        
        'concept_question':
            ## {concept}
            {explanation}
            
            **In Practice:**
            {practical_example}
            
            **Related Functions:** {function_list}
        """
        if not results:
            return f"No results found for: '{query}'"
        
        formatted = []
        
        if query_intent == 'function_lookup':
            formatted.append(f"## Function Reference for: '{query}'\n")
            for r in results[:3]:  # Limit to top 3 for function lookups
                name = r.get('function_name', r.get('name', 'Unknown'))
                pattern = r.get('calling_pattern', r.get('signature', ''))
                usage = r.get('usage_instruction', '')
                params = r.get('parameters', '')
                desc = r.get('description', '')
                
                formatted.append(f"### {name}")
                if pattern:
                    formatted.append(f"**Calling Pattern:** `{pattern}`")
                if usage:
                    formatted.append(f"**Usage:** {usage}")
                if desc:
                    formatted.append(f"**Description:** {desc}")
                if params:
                    formatted.append(f"**Parameters:** {params}")
                formatted.append("")
                
        elif query_intent == 'example_request':
            formatted.append(f"## Code Example for: '{query}'\n")
            # Show the best code example immediately
            if results:
                best = results[0]
                content = best.get('content', '')
                summary = best.get('summary', '')
                url = best.get('url', '')
                
                formatted.append("```matlab")
                formatted.append(content[:5000] if len(content) > 5000 else content)
                formatted.append("```")
                
                if summary:
                    formatted.append(f"\n**Key Points:** {summary[:200]}")
                
                functions = self._extract_functions_used(content)
                if functions:
                    formatted.append(f"**Functions Used:** {', '.join(functions[:10])}")
                
                if url:
                    formatted.append(f"**Source:** {url}")
                    
        elif query_intent == 'debug_request':
            formatted.append(f"## Debugging Help for: '{query}'\n")
            
            # Check for common errors
            if "mr.write" in query.lower():
                formatted.append("### Issue Found: Incorrect Function Call")
                formatted.append("\n**Problem:** `write` is a Sequence class method, not an mr function")
                formatted.append("**Solution:** Use `seq.write()` instead of `mr.write()`")
                formatted.append("\n**Corrected Code:**")
                formatted.append("```matlab")
                formatted.append("% Create sequence object first")
                formatted.append("seq = mr.Sequence();")
                formatted.append("")
                formatted.append("% ... your sequence code here ...")
                formatted.append("")
                formatted.append("% Save the sequence")
                formatted.append("seq.write('filename.seq');")
                formatted.append("```")
            else:
                # Generic debug help
                formatted.append("Let me help you debug this issue.")
                if results:
                    formatted.append("\nRelevant documentation:")
                    for r in results[:2]:
                        formatted.append(f"- {r.get('summary', '')[:200]}")
                        
        elif query_intent == 'tutorial_request':
            formatted.append(f"## Tutorial: {query}\n")
            formatted.append("### Understanding the Concept")
            
            if results:
                # Use the best result for tutorial content
                best = results[0]
                content = best.get('content', '')
                summary = best.get('summary', '')
                
                formatted.append(summary[:500] if summary else "")
                formatted.append("\n### Step-by-Step Implementation")
                
                # If it's a notebook, process it for tutorial
                if '.ipynb' in best.get('url', ''):
                    processed = self.process_notebook_content(content, 'tutorial_request')
                    formatted.append(processed)
                else:
                    formatted.append("```matlab")
                    formatted.append(content[:3000] if len(content) > 3000 else content)
                    formatted.append("```")
                    
        else:  # concept_question
            formatted.append(f"## Concept: {query}\n")
            if results:
                best = results[0]
                formatted.append(best.get('summary', best.get('content', ''))[:1000])
                
                formatted.append("\n**In Practice:**")
                # Try to find a practical example
                for r in results[1:3]:
                    if 'example' in r.get('summary', '').lower():
                        formatted.append(r.get('summary', '')[:300])
                        break
                
                # Extract related functions
                all_functions = set()
                for r in results[:3]:
                    functions = self._extract_functions_used(r.get('content', ''))
                    all_functions.update(functions)
                
                if all_functions:
                    formatted.append(f"\n**Related Functions:** {', '.join(list(all_functions)[:10])}")
        
        return '\n'.join(formatted)

    async def perform_rag_query(
        self,
        query: str,
        search_type: str = "auto",
        match_count: int = 10
    ) -> str:
        """
        Enhanced main RAG query with intelligent routing.
        
        Flow:
        1. Classify query intent
        2. Route to appropriate search method:
           - function_lookup → search_api_functions_enhanced
           - example_request → search_code_implementations
           - debug_request → validate_pulseq_code + search
           - tutorial_request → search with notebook processing
           - concept_question → search documentation
        3. Format results adaptively
        4. Return formatted response
        
        Ensure MATLAB is default unless Python explicitly requested.
        """
        # Start performance monitoring
        context = self.performance_monitor.start_query(query, "rag_enhanced")
        
        try:
            # Classify query intent
            intent_analysis = self.classify_query_intent(query)
            intent = intent_analysis['intent']
            language = intent_analysis.get('language', 'matlab')
            confidence = intent_analysis.get('confidence', 0.5)
            search_strategy = intent_analysis.get('search_strategy', 'unified')
            
            context["intent"] = intent
            context["confidence"] = confidence
            context["strategy"] = search_strategy
            
            logger.info(f"Query intent: {intent} (confidence: {confidence:.2f}, strategy: {search_strategy})")
            
            results = []
            
            # Route based on intent
            if search_strategy == 'api_enhanced' or intent == 'function_lookup':
                # Use enhanced API search
                api_results = await self.search_api_functions_enhanced(query, language, match_count)
                return api_results  # Already formatted
                
            elif search_strategy == 'code_search' or intent == 'example_request':
                # Search for code implementations
                code_results = self.search_code_implementations(query, language, match_count)
                return code_results  # Already formatted
                
            elif search_strategy == 'debug_validate' or intent == 'debug_request':
                # First check if user provided code to validate
                if '```' in query:  # User provided code block
                    # Extract code from markdown
                    import re
                    code_match = re.search(r'```(?:matlab|python)?\n(.*?)\n```', query, re.DOTALL)
                    if code_match:
                        user_code = code_match.group(1)
                        validation = self.validate_pulseq_code(user_code, language)
                        
                        # Format validation results
                        formatted = ["## Code Validation Results\n"]
                        
                        if validation['errors']:
                            formatted.append("### ❌ Errors Found:")
                            for error in validation['errors']:
                                formatted.append(f"\n**Issue:** {error['error']}")
                                formatted.append(f"**Found:** `{error['found_usage']}`")
                                formatted.append(f"**Correct:** `{error['correct_usage']}`")
                                formatted.append(f"**Fix:** {error['fix']}")
                        
                        if validation['suggestions']:
                            formatted.append("\n### 💡 Suggestions:")
                            for suggestion in validation['suggestions']:
                                formatted.append(f"- {suggestion}")
                        
                        if validation['valid_functions']:
                            formatted.append(f"\n### ✅ Valid Functions: {', '.join(validation['valid_functions'][:10])}")
                        
                        return '\n'.join(formatted)
                else:
                    # Search for debug help
                    results = self._search_debug_help(query, language, match_count)
                    return self.format_results_adaptive(results, intent, query)
                    
            elif search_strategy == 'tutorial_search' or intent == 'tutorial_request':
                # Search for tutorials with notebook priority
                results = self._search_tutorials(query, language, match_count)
                return self.format_results_adaptive(results, intent, query)
                
            else:  # documentation or unified
                # Standard documentation search
                # Use existing perform_rag_query logic but simplified
                search_limit = min(match_count * 3, 100)
                if self.settings.use_hybrid_search:
                    results = self.supabase_client.perform_hybrid_search(
                        query=query,
                        match_count=search_limit,
                        search_type="documents",
                    )
                else:
                    results = self.supabase_client.search_documents(
                        query=query,
                        match_count=search_limit,
                    )
                
                # Limit and format
                results = results[:match_count] if results else []
                return self.format_results_adaptive(results, intent, query)
                
        except Exception as e:
            # Record failure
            self.performance_monitor.record_query_completion(context, [], error=str(e))
            logger.error(f"Enhanced RAG query failed: {e}")
            # Fall back to simpler search
            return self._perform_simple_search(query, match_count)
    
    def _search_debug_help(self, query: str, language: str, match_count: int) -> List[Dict]:
        """Search for debugging help and solutions."""
        # Search for error-related content
        try:
            query_builder = self.supabase_client.client.from_("crawled_pages").select("*")
            query_builder = query_builder.or_(
                f"content.ilike.%error%,"
                f"content.ilike.%debug%,"
                f"content.ilike.%troubleshoot%,"
                f"summary.ilike.%{query}%"
            )
            result = query_builder.limit(match_count).execute()
            return result.data if result.data else []
        except Exception:
            return []
    
    def _search_tutorials(self, query: str, language: str, match_count: int) -> List[Dict]:
        """Search for tutorial content with notebook priority."""
        try:
            query_builder = self.supabase_client.client.from_("crawled_pages").select("*")
            # Prioritize notebooks and tutorial content
            query_builder = query_builder.or_(
                "metadata->>file_extension.eq..ipynb,"
                "url.ilike.%tutorial%,"
                "url.ilike.%example%,"
                "summary.ilike.%tutorial%"
            )
            query_builder = query_builder.or_(f"summary.ilike.%{query}%")
            result = query_builder.limit(match_count).execute()
            return result.data if result.data else []
        except Exception:
            return []
    
    def _perform_simple_search(self, query: str, match_count: int) -> str:
        """Fallback simple search when enhanced search fails."""
        try:
            # Just do a basic search
            results = self.supabase_client.search_documents(
                query=query,
                match_count=match_count
            )
            return self._format_rag_results(results, query)
        except Exception:
            return "Unable to search at this time. Please try again."

    def perform_rag_query(  # noqa: F811
        self,
        query: str,
        source: Optional[str] = None,
        match_count: int = 5,
        use_hybrid: bool = True,
    ) -> str:
        """
        Search the RAG database for documentation and information.
        
        NOTE: This is the sync version - async version exists at line 1232.
        Both are kept for backward compatibility with tools.py executor pattern.

        Args:
            query: Search query
            source: Optional source filter
            match_count: Number of results to return
            use_hybrid: Whether to use hybrid search (vector + keyword)

        Returns:
            Formatted search results
        """
        # Start performance monitoring
        context = self.performance_monitor.start_query(query, "documents")
        context["hybrid_search"] = use_hybrid

        try:
            # Get sufficient results for good ranking without being excessive
            search_limit = min(match_count * 10, 200)  # Get 10x requested or max 200
            if use_hybrid and self.settings.use_hybrid_search:
                results = self.supabase_client.perform_hybrid_search(
                    query=query,
                    match_count=search_limit,
                    source=source,
                    search_type="documents",
                )
            else:
                filter_metadata = {"source": source} if source else None
                results = self.supabase_client.search_documents(
                    query=query,
                    match_count=search_limit,
                    filter_metadata=filter_metadata,
                )

            # After getting all results with proper ranking, limit to requested display count
            if results and len(results) > match_count:
                results = results[:match_count]

            # Record successful completion
            self.performance_monitor.record_query_completion(context, results)

            # Format results
            return self._format_rag_results(results, query)

        except Exception as e:
            # Record failure
            self.performance_monitor.record_query_completion(context, [], error=str(e))
            logger.error(f"RAG query failed: {e}")
            return f"Error performing RAG search: {str(e)}"

    def search_code_implementations(self, query: str, language: str = "matlab", match_count: int = 10) -> str:
        """
        Search for code implementations in crawled_pages.
        
        Steps:
        1. Determine file extensions based on language:
           - MATLAB: [".m", ".mlx", ".ipynb"]
           - Python: [".py", ".ipynb"]
        
        2. Search crawled_pages with filter:
           - Use metadata->file_extension filter
           - Search both summary and content
           - Prioritize files with seq.write() (complete sequences)
        
        3. Process results based on file type:
           - Regular code files: Return as-is
           - Notebooks: Process based on query intent
        
        4. Format results with:
           - Full code
           - Brief explanation from summary
           - Source attribution
        
        Returns formatted code examples ready for use.
        """
        context = self.performance_monitor.start_query(query, "code_implementations")
        context["language"] = language
        
        try:
            # Determine file extensions based on language
            if language.lower() == "python":
                extensions = [".py", ".ipynb"]
            else:  # Default to MATLAB
                extensions = [".m", ".mlx", ".ipynb"]
            
            # Build query for crawled_pages
            query_builder = self.supabase_client.client.from_("crawled_pages").select("*")
            
            # Filter by file extension using metadata JSONB column
            extension_filters = []
            for ext in extensions:
                extension_filters.append(f"metadata->>file_extension.eq.{ext}")
            
            if extension_filters:
                query_builder = query_builder.or_(",".join(extension_filters))
            
            # Add text search
            query_builder = query_builder.or_(
                f"summary.ilike.%{query}%,"
                f"content.ilike.%{query}%,"
                f"url.ilike.%{query}%"
            )
            
            # Execute query
            result = query_builder.limit(match_count * 2).execute()  # Get extra for filtering
            results = result.data if result.data else []
            
            # Prioritize results: 1) MATLAB with seq.write, 2) MATLAB partial, 3) Python with seq.write, 4) Python partial
            matlab_complete = []
            matlab_partial = []
            python_complete = []
            python_partial = []
            other_results = []
            
            for r in results:
                content = r.get("content", "")
                metadata = r.get("metadata", {})
                url = r.get("url", "").lower()
                
                # Check if complete sequence
                is_complete = "seq.write(" in content or "seq.write (" in content or "seq.write('" in content
                
                # Determine language
                is_matlab = (
                    metadata.get("language", "").lower() == "matlab" or
                    metadata.get("file_extension", "") == ".m" or
                    ".m" in url or
                    "/matlab/" in url or
                    "mr.Sequence()" in content  # MATLAB pattern
                )
                
                is_python = (
                    metadata.get("language", "").lower() == "python" or
                    metadata.get("file_extension", "") == ".py" or
                    ".py" in url or
                    "/python/" in url or
                    "pypulseq" in url or
                    "import pypulseq" in content or
                    "pp.Sequence()" in content  # Python pattern
                )
                
                # Categorize results
                if is_matlab:
                    if is_complete:
                        matlab_complete.append(r)
                    else:
                        matlab_partial.append(r)
                elif is_python:
                    if is_complete:
                        python_complete.append(r)
                    else:
                        python_partial.append(r)
                else:
                    other_results.append(r)
            
            # Combine with MATLAB priority (unless Python explicitly requested)
            if language.lower() == "python":
                # Python explicitly requested - still show Python first
                prioritized_results = python_complete + python_partial + matlab_complete + matlab_partial + other_results
            else:
                # Default to MATLAB priority
                prioritized_results = matlab_complete + matlab_partial + other_results + python_complete + python_partial
                
                if matlab_complete or matlab_partial:
                    logger.info(f"Prioritized {len(matlab_complete) + len(matlab_partial)} MATLAB results")
                elif python_complete or python_partial:
                    logger.info(f"No MATLAB results found, showing {len(python_complete) + len(python_partial)} Python results")
            
            # Limit to requested count
            final_results = prioritized_results[:match_count]
            
            # Format results
            if not final_results:
                return f"No code implementations found for: '{query}'"
            
            formatted = [f"## Code Implementations for: '{query}'\n"]
            
            # Check what language results we're actually showing
            first_result_lang = None
            if final_results:
                first_r = final_results[0]
                first_metadata = first_r.get("metadata", {})
                first_url = first_r.get("url", "").lower()
                first_content = first_r.get("content", "")
                
                if (first_metadata.get("language", "").lower() == "python" or
                    ".py" in first_url or "pypulseq" in first_url or
                    "import pypulseq" in first_content):
                    first_result_lang = "python"
                elif (first_metadata.get("language", "").lower() == "matlab" or
                      ".m" in first_url or "mr.Sequence()" in first_content):
                    first_result_lang = "matlab"
            
            # Add language note
            if language.lower() == "matlab" and first_result_lang == "python":
                formatted.append("*Note: Showing Python implementation (no MATLAB version found). The MATLAB equivalent would use similar logic with mr.* functions.*\n")
            elif language.lower() == "python" and first_result_lang == "matlab":
                formatted.append("*Note: Showing MATLAB implementation. For Python/pypulseq, adapt using pp.* functions.*\n")
            else:
                formatted.append(f"*Language: {(first_result_lang or language).upper()}*\n")
            
            formatted.append(f"Found {len(final_results)} implementation(s):\n")
            
            for i, item in enumerate(final_results, 1):
                metadata = item.get("metadata", {})
                file_ext = metadata.get("file_extension", "")
                summary = item.get("summary", "No description available")
                content = item.get("content", "")
                url = item.get("url", "N/A")
                
                # Process notebooks if needed
                if file_ext == ".ipynb" and "```" not in content:
                    content = self.process_notebook_content(content, "example_request")
                
                formatted.append(f"### {i}. Implementation from {url.split('/')[-1]}")
                formatted.append(f"**Summary:** {summary[:200]}...")
                formatted.append(f"**Source:** {url}")
                
                # Determine code language for syntax highlighting
                code_lang = "python" if file_ext == ".py" else "matlab"
                
                formatted.append(f"\n```{code_lang}")
                # Limit code preview if too long
                if len(content) > 5000:
                    formatted.append(content[:5000])
                    formatted.append(f"\n% ... [Truncated - {len(content)} total characters]")
                else:
                    formatted.append(content)
                formatted.append("```")
                
                # Add key functions used
                functions_used = self._extract_functions_used(content)
                if functions_used:
                    formatted.append(f"**Key Functions:** {', '.join(functions_used[:10])}")
                
                formatted.append("")  # Empty line between results
            
            self.performance_monitor.record_query_completion(context, final_results)
            return "\n".join(formatted)
            
        except Exception as e:
            self.performance_monitor.record_query_completion(context, [], error=str(e))
            logger.error(f"Code implementation search failed: {e}")
            return f"Error searching code implementations: {str(e)}"
    
    def _extract_functions_used(self, code: str) -> List[str]:
        """Extract Pulseq functions used in code."""
        import re
        
        # Patterns for common Pulseq functions
        patterns = [
            r'mr\.(\w+)\(',  # mr.functionName(
            r'seq\.(\w+)\(',  # seq.methodName(
            r'make(\w+)\(',  # makeTrapezoid(
            r'calc(\w+)\(',  # calcDuration(
            r'write(\w+)\(',  # writeHASTE(
        ]
        
        functions = set()
        for pattern in patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            for match in matches:
                if pattern.startswith('mr'):
                    functions.add(f"mr.{match}")
                elif pattern.startswith('seq'):
                    functions.add(f"seq.{match}")
                else:
                    functions.add(match.lower())
        
        return sorted(list(functions))
    
    def validate_pulseq_code(self, code: str, language: str = "matlab") -> Dict[str, Any]:
        """
        Validate Pulseq function calls in user code.
        
        Steps:
        1. Extract all function calls matching:
           - mr.* patterns
           - seq.* patterns
           - Common sequence operations
        
        2. For each function found:
           - Query function_calling_patterns view
           - Check if calling pattern is correct
           - Identify class methods used as regular functions
        
        3. Return validation results:
           - 'valid_functions': List of correctly used functions
           - 'errors': List of incorrect usage with corrections
           - 'warnings': Potential issues (deprecated, etc.)
           - 'suggestions': Improvements based on best practices
        
        Example error:
        {
            'function': 'write',
            'found_usage': 'mr.write(...)',
            'correct_usage': 'seq.write(...)',
            'fix': 'This is a Sequence class method. First create: seq = mr.Sequence();'
        }
        """
        import re
        
        validation_results = {
            'valid_functions': [],
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Extract function calls
        function_patterns = [
            (r'mr\.(\w+)\s*\(', 'mr'),
            (r'seq\.(\w+)\s*\(', 'seq'),
            (r'(\w+)\s*=\s*make(\w+)\s*\(', 'make'),
            (r'(\w+)\s*=\s*calc(\w+)\s*\(', 'calc'),
            (r'write(\w+)\s*\(', 'write')
        ]
        
        found_functions = []
        for pattern, prefix in function_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                if prefix == 'mr':
                    func_name = match.group(1)
                    found_functions.append(('mr', func_name, match.start()))
                elif prefix == 'seq':
                    func_name = match.group(1)
                    found_functions.append(('seq', func_name, match.start()))
                elif prefix in ['make', 'calc', 'write']:
                    func_name = prefix + match.group(2) if len(match.groups()) > 1 else prefix + match.group(1)
                    found_functions.append(('function', func_name, match.start()))
        
        # Common errors database
        common_errors = {
            'mr.write': {
                'error': 'write is a Sequence class method, not an mr function',
                'correct': 'seq.write',
                'fix': 'First create sequence: seq = mr.Sequence(); then use seq.write("filename.seq")'
            },
            'mr.addBlock': {
                'error': 'addBlock is a Sequence class method',
                'correct': 'seq.addBlock',
                'fix': 'Use seq.addBlock(...) after creating sequence object'
            },
            'mr.plot': {
                'error': 'plot is a Sequence class method',
                'correct': 'seq.plot',
                'fix': 'Use seq.plot() after creating sequence object'
            },
            'mr.setDefinition': {
                'error': 'setDefinition is a Sequence class method',
                'correct': 'seq.setDefinition',
                'fix': 'Use seq.setDefinition(...) after creating sequence object'
            }
        }
        
        # Validate each function
        for prefix, func_name, position in found_functions:
            full_name = f"{prefix}.{func_name}" if prefix != 'function' else func_name
            
            # Check common errors
            if full_name in common_errors:
                error_info = common_errors[full_name]
                validation_results['errors'].append({
                    'function': func_name,
                    'found_usage': full_name,
                    'correct_usage': error_info['correct'],
                    'error': error_info['error'],
                    'fix': error_info['fix'],
                    'position': position
                })
            else:
                # Mark as valid (could enhance with actual database lookup)
                validation_results['valid_functions'].append(full_name)
        
        # Add suggestions based on best practices
        if 'seq = mr.Sequence()' not in code and 'Sequence()' not in code:
            if any('seq.' in f for f, _, _ in found_functions):
                validation_results['suggestions'].append(
                    "Consider adding 'seq = mr.Sequence()' at the beginning of your script"
                )
        
        if 'seq.write' not in code and 'write(' not in code:
            validation_results['suggestions'].append(
                "Don't forget to save your sequence with seq.write('filename.seq')"
            )
        
        if 'seq.plot' not in code:
            validation_results['suggestions'].append(
                "Consider adding seq.plot() to visualize your sequence"
            )
        
        return validation_results

    def process_notebook_content(self, notebook_content: str, query_intent: str) -> str:
        """
        Process Jupyter notebook content based on user intent.
        
        For 'example_request':
        - Extract only code cells
        - Remove markdown cells
        - Concatenate code into runnable script
        - Add comment with source notebook
        
        For 'tutorial_request':
        - Preserve markdown explanations
        - Keep code cells in sequence
        - Format as educational progression
        - Maintain step-by-step narrative
        
        Handle both .ipynb JSON structure and processed content.
        """
        import json
        
        try:
            # Check if content is JSON (raw notebook)
            if notebook_content.strip().startswith('{'):
                try:
                    notebook = json.loads(notebook_content)
                    cells = notebook.get('cells', [])
                except json.JSONDecodeError:
                    # Not valid JSON, treat as processed content
                    return notebook_content
            else:
                # Already processed content, return as-is
                return notebook_content
            
            if query_intent == "example_request":
                # Extract only code cells for runnable script
                code_blocks = []
                code_blocks.append("% Extracted from Jupyter notebook")
                code_blocks.append("% Combined code cells for direct execution\n")
                
                for cell in cells:
                    if cell.get('cell_type') == 'code':
                        source = cell.get('source', [])
                        if isinstance(source, list):
                            code = ''.join(source)
                        else:
                            code = source
                        
                        # Skip empty cells
                        if code.strip():
                            code_blocks.append(code)
                            code_blocks.append("")  # Add spacing
                
                return '\n'.join(code_blocks)
                
            elif query_intent == "tutorial_request":
                # Preserve educational structure
                formatted_content = []
                
                for i, cell in enumerate(cells):
                    cell_type = cell.get('cell_type')
                    source = cell.get('source', [])
                    
                    if isinstance(source, list):
                        content = ''.join(source)
                    else:
                        content = source
                    
                    if cell_type == 'markdown':
                        # Preserve markdown for context
                        formatted_content.append("% === Explanation ===")
                        # Convert markdown to comments
                        for line in content.split('\n'):
                            if line.strip():
                                formatted_content.append(f"% {line}")
                        formatted_content.append("")
                        
                    elif cell_type == 'code' and content.strip():
                        formatted_content.append(f"% === Code Block {i+1} ===")
                        formatted_content.append(content)
                        formatted_content.append("")
                
                return '\n'.join(formatted_content)
            
            else:
                # Default: return code cells with minimal context
                return self.process_notebook_content(notebook_content, "example_request")
                
        except Exception as e:
            logger.warning(f"Error processing notebook content: {e}")
            # Return original content if processing fails
            return notebook_content

    async def search_code_examples(
        self,
        query: str,
        source_id: Optional[str] = None,
        match_count: int = 5,
        use_hybrid: bool = True,
    ) -> str:
        """
        Search for code examples with priority for official sequences.
        Searches official Pulseq sequences first, then falls back to other examples.

        Args:
            query: Code search query
            source_id: Optional source ID filter
            match_count: Number of results to return
            use_hybrid: Whether to use hybrid search

        Returns:
            Formatted code examples
        """
        # Start performance monitoring
        context = self.performance_monitor.start_query(query, "code_examples")

        try:
            # REMOVED: Special handling for official sequences was returning full files (21KB)
            # instead of search snippets, causing RECITATION errors
            # Let the regular search path handle sequence requests like it did before
            
            # Continue with regular search that returns snippets
            # Classify the query to determine search strategy
            strategy, metadata = self.classify_search_strategy(query)
            context["search_strategy"] = strategy
            
            # Get appropriate number of results based on strategy
            if strategy == "vector_enhanced":
                initial_match_count = 50  # Reduced for better performance
            else:
                initial_match_count = min(match_count * 3, 100)
            
            if strategy == "vector_enhanced":
                # Enhanced search for MRI sequences
                seq_type = metadata.get("sequence_type", "")
                
                # Try to find files with the sequence name in the URL/filename first
                # This often gives the best results
                if use_hybrid and self.settings.use_hybrid_search:
                    # Use hybrid search but focus on sequence-specific keywords
                    results = self.supabase_client.perform_hybrid_search(
                        query=query,
                        match_count=initial_match_count,
                        source=source_id,
                        search_type="code_examples",
                        keyword_query_override=seq_type  # Just search for the sequence type
                    )
                else:
                    filter_metadata = {"source": source_id} if source_id else None
                    results = self.supabase_client.search_code_examples(
                        query=query,
                        match_count=initial_match_count,
                        filter_metadata=filter_metadata,
                        source_id=source_id,
                        use_reranking=True
                    )
                
                # Use the new scoring method for sequences
                if results:
                    scored_results = []
                    for result in results:
                        # Use the new _score_sequence_relevance method
                        score = self._score_sequence_relevance(result, seq_type)
                        
                        # Only keep results with positive scores or ensure minimum results
                        if score > 0 or len(scored_results) < 10:
                            result['relevance_score'] = score
                            scored_results.append(result)
                    
                    # Sort by relevance score
                    scored_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
                    results = scored_results
                
            elif strategy == "hybrid_filtered":
                # Use filtered query for keyword portion
                filtered_query = metadata.get("filtered_query", query)
                results = self.supabase_client.perform_hybrid_search(
                    query=query,  # Original for vector search
                    match_count=initial_match_count,
                    source=source_id,
                    search_type="code_examples",
                    keyword_query_override=filtered_query  # Filtered for keywords
                )
                
            else:  # hybrid_full
                # Standard hybrid search
                results = self.supabase_client.perform_hybrid_search(
                    query=query,
                    match_count=initial_match_count,
                    source=source_id,
                    search_type="code_examples"
                )
            
            # Apply MATLAB preference if no language explicitly specified
            query_lower = query.lower()
            if results and "python" not in query_lower and "pypulseq" not in query_lower:
                # Sort results to prioritize MATLAB examples
                # Check multiple fields for language identification
                matlab_results = []
                python_results = []
                other_results = []
                
                for r in results:
                    # Check metadata, url, and content for language indicators
                    metadata = r.get("metadata", {})
                    url = r.get("url", "").lower()
                    content = r.get("content", "").lower()
                    
                    # Determine language from multiple sources
                    is_matlab = (
                        metadata.get("language", "").lower() == "matlab" or
                        ".m" in url or
                        "/matlab/" in url or
                        "% matlab" in content[:100] or
                        "mr.Sequence()" in r.get("content", "")  # MATLAB pattern
                    )
                    
                    is_python = (
                        metadata.get("language", "").lower() == "python" or
                        ".py" in url or
                        "/python/" in url or
                        "pypulseq" in url or
                        "import pypulseq" in content or
                        "pp.Sequence()" in r.get("content", "")  # Python pattern
                    )
                    
                    if is_matlab:
                        matlab_results.append(r)
                    elif is_python:
                        python_results.append(r)
                    else:
                        other_results.append(r)
                
                # ALWAYS put MATLAB results first if they exist
                if matlab_results:
                    results = matlab_results + other_results + python_results
                    logger.info(f"Prioritized {len(matlab_results)} MATLAB results over {len(python_results)} Python results")
                elif not python_results:
                    # No language-specific results, keep original order
                    pass
                else:
                    # Only Python results found - still put them last and other results first
                    results = other_results + python_results
                    logger.info(f"No MATLAB results found, showing {len(python_results)} Python results")
            
            # Limit to requested count
            results = results[:match_count] if results else []
            
            # Record completion
            self.performance_monitor.record_query_completion(context, results)
            
            # Format results
            return self._format_code_results(results, query)

        except Exception as e:
            # Record failure
            self.performance_monitor.record_query_completion(context, [], error=str(e))
            logger.error(f"Code search failed: {e}")
            return f"Error searching code examples: {str(e)}"

    def get_available_sources(self) -> str:
        """
        Get list of available documentation sources.

        Returns:
            Formatted list of sources
        """
        try:
            sources = self.supabase_client.get_available_sources()
            return self._format_sources_results(sources)

        except Exception as e:
            logger.error(f"Sources query failed: {e}")
            return f"Error retrieving sources: {str(e)}"

    def _format_rag_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Format RAG search results for display."""
        if not results:
            return f"No documentation found for query: '{query}'"

        formatted = [f"## Documentation Results for: '{query}'\n"]
        formatted.append(f"Found {len(results)} relevant documents:\n")

        for i, item in enumerate(results, 1):
            # Extract metadata
            metadata = item.get("metadata", {})
            headers = metadata.get("headers", "")

            # Extract content - use configurable limit for AI context
            content = item.get("content", "")

            # Only truncate if content is extremely long
            if len(content) > self.doc_preview_limit:
                content_preview = (
                    content[: self.doc_preview_limit]
                    + f"\n... [Truncated - {len(content)} total characters]"
                )
            else:
                content_preview = content

            # Format result
            formatted.append(
                f"### {i}. Result from {item.get('source_id', 'Unknown source')}"
            )
            if headers:
                formatted.append(f"**Section:** {headers}")
            formatted.append(f"**URL:** {item.get('url', 'N/A')}")
            formatted.append(f"**Relevance:** {item.get('similarity', 0):.2%}")
            formatted.append(f"\n{content_preview}")
            formatted.append("")  # Empty line between results

        return "\n".join(formatted)

    def _format_code_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Format code search results for display."""
        if not results:
            return f"No code examples found for query: '{query}'"

        formatted = [f"## Code Examples for: '{query}'\n"]
        
        # Add language preference note if applicable
        query_lower = query.lower()
        if results:
            # Detect the actual language of the first result
            first_result = results[0]
            first_metadata = first_result.get("metadata", {})
            first_url = first_result.get("url", "").lower()
            first_content = first_result.get("content", "")
            
            is_python_result = (
                first_metadata.get("language", "").lower() == "python" or
                ".py" in first_url or
                "pypulseq" in first_url or
                "import pypulseq" in first_content[:200]
            )
            
            is_matlab_result = (
                first_metadata.get("language", "").lower() == "matlab" or
                ".m" in first_url or
                "mr.Sequence()" in first_content[:200]
            )
            
            # Add appropriate note based on what we're showing vs what was requested
            if "python" not in query_lower and "pypulseq" not in query_lower:
                # User didn't ask for Python
                if is_python_result:
                    formatted.append("*Note: Showing Python example (no MATLAB version found). The MATLAB equivalent would use mr.* functions instead of pp.*.*\n")
                elif is_matlab_result:
                    formatted.append("*Language: MATLAB (default). Add 'python' or 'pypulseq' to your query for Python examples.*\n")
            elif "python" in query_lower or "pypulseq" in query_lower:
                # User asked for Python
                if is_matlab_result:
                    formatted.append("*Note: Showing MATLAB example (no Python version found). Adapt using pypulseq's pp.* functions.*\n")
                elif is_python_result:
                    formatted.append("*Language: Python (pypulseq) as requested.*\n")
        
        # Only show first result to prevent RECITATION errors from large content
        results_to_show = results[:1]  # Just the top result
        if len(results) > 1:
            formatted.append(f"Found {len(results)} code examples:\n")
        else:
            formatted.append(f"Found {len(results)} code example:\n")

        for i, item in enumerate(results_to_show, 1):
            # Extract metadata
            metadata = item.get("metadata", {})
            language = metadata.get("language", "Unknown")

            # Get summary and full code
            summary = item.get("summary", "No description available")
            code = item.get("content", "")

            # Only truncate if code is extremely long
            if len(code) > self.code_preview_limit:
                code_preview = (
                    code[: self.code_preview_limit]
                    + f"\n% ... [Truncated - {len(code)} total characters]"
                )
            else:
                code_preview = code

            # Format result
            formatted.append(
                f"### {i}. {language.upper()} Example from {item.get('source_id', 'Unknown')}"
            )
            formatted.append(f"**Summary:** {summary}")
            formatted.append(f"**URL:** {item.get('url', 'N/A')}")
            formatted.append(f"**Relevance:** {item.get('similarity', 0):.2%}")
            formatted.append(f"\n```{language}")
            formatted.append(code_preview)
            formatted.append("```")
            formatted.append("")  # Empty line between results

        return "\n".join(formatted)

    def format_code_results_interactive(
        self, results: List[Dict[str, Any]], query: str
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Format code search results - always show the top result directly.
        
        Returns:
            tuple: (formatted_string, None) - no longer returns pending results
        """
        if not results:
            return f"No code examples found for query: '{query}'", None

        # Always show the top result directly
        top_result = results[:1]  # Just the best match
        
        # Add a note if there were multiple results
        if len(results) > 1:
            formatted_result = f"## Found {len(results)} implementations for: '{query}'\n"
            formatted_result += "Showing the most relevant result:\n\n"
            formatted_result += self._format_code_results(top_result, query)
        else:
            # Single result - show without the "found X implementations" header
            formatted_result = self._format_code_results(top_result, query)
        
        return formatted_result, None  # Always return None for pending results

    def format_selected_code_results(
        self, results: List[Dict[str, Any]], selection, query: str
    ) -> str:
        """Format selected code results based on user choice (1-3 or all)."""
        if selection == "all":
            return self._format_code_results(results, query)  # Show all (max 3)

        try:
            index = int(selection) - 1
            if 0 <= index < len(results):
                return self._format_code_results([results[index]], query)
            else:
                max_num = min(3, len(results))
                if max_num == 1:
                    return "Invalid selection. This is the only result available."
                elif max_num == 2:
                    return "Invalid selection. Please choose 1, 2, or 'all'."
                else:
                    return "Invalid selection. Please choose a number between 1 and 3, or 'all'."
        except (ValueError, TypeError):
            max_num = min(3, len(results))
            if max_num == 2:
                return "Please reply with a number (1 or 2) or 'all'."
            else:
                return "Please reply with a number (1, 2, or 3) or 'all'."

    def _format_sources_results(self, sources: List[Dict[str, Any]]) -> str:
        """Format available sources for display."""
        if not sources:
            return "No sources found in the database."

        formatted = ["## Available Documentation Sources\n"]
        formatted.append(f"Total sources in database: {len(sources)}\n")

        # Group sources by type
        matlab_sources = []
        python_sources = []
        tutorial_sources = []
        other_sources = []

        for source in sources:
            source_id = source.get("source_id", "Unknown")
            summary = source.get("summary", "No description available")
            total_words = source.get("total_words", 0)

            # Create source entry
            entry = f"- **{source_id}** ({total_words:,} words)\n  {summary}"

            # Categorize by source ID patterns
            if "matlab" in source_id.lower() or "pulseq/pulseq" in source_id:
                matlab_sources.append(entry)
            elif "python" in source_id.lower() or "pypulseq" in source_id:
                python_sources.append(entry)
            elif "tutorial" in source_id.lower():
                tutorial_sources.append(entry)
            else:
                other_sources.append(entry)

        # Add categorized sources
        if matlab_sources:
            formatted.append("### MATLAB/Octave Sources:")
            formatted.extend(matlab_sources)
            formatted.append("")

        if python_sources:
            formatted.append("### Python Sources:")
            formatted.extend(python_sources)
            formatted.append("")

        if tutorial_sources:
            formatted.append("### Tutorial Sources:")
            formatted.extend(tutorial_sources)
            formatted.append("")

        if other_sources:
            formatted.append("### Other Sources:")
            formatted.extend(other_sources)
            formatted.append("")

        return "\n".join(formatted)

    def get_performance_stats(self, window_minutes: Optional[int] = None) -> str:
        """
        Get RAG performance statistics.

        Args:
            window_minutes: Optional time window for statistics

        Returns:
            Formatted performance statistics
        """
        try:
            stats = self.performance_monitor.get_performance_stats(window_minutes)
            duration_percentiles = self.performance_monitor.get_percentiles("duration")
            query_patterns = self.performance_monitor.get_query_pattern_analysis()

            formatted = ["## RAG Performance Statistics\n"]

            if window_minutes:
                formatted.append(f"**Time Window:** Last {window_minutes} minutes\n")
            else:
                formatted.append("**Time Window:** All time\n")

            # Basic stats
            formatted.extend(
                [
                    f"**Total Queries:** {stats.total_queries}",
                    f"**Average Duration:** {stats.avg_duration:.3f}s",
                    f"**Min/Max Duration:** {stats.min_duration:.3f}s / {stats.max_duration:.3f}s",
                    f"**Average Results:** {stats.avg_result_count:.1f}",
                    f"**Average Similarity:** {stats.avg_similarity:.3f}",
                    f"**Error Rate:** {stats.error_rate:.1%}",
                    f"**Queries per Second:** {stats.queries_per_second:.2f}",
                    f"**Cache Hit Rate:** {stats.cache_hit_rate:.1%}",
                    "",
                ]
            )

            # Duration percentiles
            if duration_percentiles:
                formatted.append("### Response Time Percentiles")
                for percentile, value in duration_percentiles.items():
                    formatted.append(f"**{percentile.upper()}:** {value:.3f}s")
                formatted.append("")

            # Query patterns
            if query_patterns:
                formatted.append("### Query Patterns")
                formatted.append(
                    f"**Search Type Distribution:** {query_patterns.get('search_type_distribution', {})}"
                )
                formatted.append(
                    f"**Average Query Length:** {query_patterns.get('avg_query_length', 0):.1f} chars"
                )
                formatted.append(
                    f"**Reranking Usage:** {query_patterns.get('reranking_usage_rate', 0):.1%}"
                )
                formatted.append(
                    f"**Hybrid Search Usage:** {query_patterns.get('hybrid_search_usage_rate', 0):.1%}"
                )
                formatted.append(
                    f"**Unique Queries:** {query_patterns.get('total_unique_queries', 0)}"
                )
                formatted.append("")

            # Slow queries
            slow_queries = self.performance_monitor.get_slow_queries(1.0, 5)
            if slow_queries:
                formatted.append("### Slowest Queries (>1s)")
                for i, q in enumerate(slow_queries, 1):
                    formatted.append(f"{i}. **{q.duration:.3f}s** - {q.query[:50]}...")
                formatted.append("")

            # Failed queries
            failed_queries = self.performance_monitor.get_failed_queries(5)
            if failed_queries:
                formatted.append("### Recent Failed Queries")
                for i, q in enumerate(failed_queries, 1):
                    formatted.append(
                        f"{i}. **Error:** {q.error} - Query: {q.query[:50]}..."
                    )
                formatted.append("")

            return "\n".join(formatted)

        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return f"Error retrieving performance statistics: {str(e)}"

    async def search_functions_fast(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Phase 1: Lightweight search for function discovery.
        Uses only essential fields for fast matching.
        
        Args:
            query: Search query for functions
            limit: Maximum number of results to return
            
        Returns:
            List of function matches with minimal fields
        """
        context = self.performance_monitor.start_query(query, "functions_fast")
        
        try:
            # Create embedding for the query
            from .embeddings import create_embedding
            query_embedding = create_embedding(query)
            
            # Query the api_reference_search view (lean fields only)
            params = {
                "query_embedding": query_embedding,
                "match_count": limit
            }
            
            # Execute search against the lean view
            result = self.supabase_client.client.rpc(
                "match_api_reference_search", params
            ).execute()
            
            if not result.data:
                # Fallback to direct query
                result = self.supabase_client.client.from_("api_reference_search").select(
                    "name, signature, description, calling_pattern, is_class_method"
                ).limit(limit).execute()
            
            self.performance_monitor.record_query_completion(context, result.data or [])
            return result.data or []
            
        except Exception as e:
            self.performance_monitor.record_query_completion(context, [], error=str(e))
            logger.error(f"Fast function search failed: {e}")
            return []
    
    async def get_function_details(self, function_name: str) -> Optional[Dict[str, Any]]:
        """
        Phase 2: Get complete details for code generation.
        Fetches heavy fields only when needed.
        
        Args:
            function_name: Exact name of the function
            
        Returns:
            Complete function details or None if not found
        """
        try:
            # Query the api_reference_details view for full details
            result = self.supabase_client.client.from_("api_reference_details").select(
                "name, signature, parameters, usage_examples, returns, has_nargin_pattern, calling_pattern"
            ).eq("name", function_name).single().execute()
            
            return result.data
            
        except Exception as e:
            logger.error(f"Failed to get function details for {function_name}: {e}")
            return None
    
    async def search_official_sequences(self, sequence_type: str = None) -> List[Dict[str, Any]]:
        """
        Search only official, validated sequence examples.
        
        Args:
            sequence_type: Optional filter for specific sequence type (e.g., 'EPI', 'SpinEcho')
            
        Returns:
            List of official sequence examples
        """
        try:
            # Query the official_sequence_examples view
            # Use ai_summary instead of content for search results
            query = self.supabase_client.client.from_("official_sequence_examples").select(
                "url, ai_summary, file_name, sequence_type"
            )
            
            # Apply sequence type filter if provided
            if sequence_type:
                query = query.eq("sequence_type", sequence_type)
            
            result = query.execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"Failed to search official sequences: {e}")
            return []


# Global instance
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """
    Get the global RAG service instance with optimizations applied.

    Returns:
        RAGService instance with performance optimizations
    """
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
        # Apply performance optimizations for faster data fetching
        try:
            from .apply_optimizations import apply_rag_optimizations
            apply_rag_optimizations()
        except Exception as e:
            logger.warning(f"Could not apply RAG optimizations: {e}")
    return _rag_service
