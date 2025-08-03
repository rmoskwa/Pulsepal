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
from .rag_performance import get_performance_monitor, monitor_query

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
        self.code_preview_limit = 10000  # Was hardcoded 300
    
    @property
    def supabase_client(self) -> SupabaseRAGClient:
        """Lazy load Supabase client."""
        if self._supabase_client is None:
            self._supabase_client = get_supabase_client()
        return self._supabase_client
    
    def classify_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Classify query to determine which tables to search.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with classification results
        """
        query_lower = query.lower()
        
        # Keywords that indicate different search types
        function_keywords = [
            'make', 'add', 'calc', 'get', 'set', 'mr.', 'seq.',
            'function', 'method', 'api', 'signature', 'parameter',
            'returns', 'arguments', 'usage'
        ]
        
        code_keywords = [
            'example', 'code', 'implement', 'write', 'create',
            'sequence', 'script', 'program', 'show me', 'how to',
            'demo', 'sample'
        ]
        
        doc_keywords = [
            'explain', 'what is', 'how does', 'why', 'when',
            'theory', 'concept', 'tutorial', 'guide', 'overview',
            'understand', 'physics', 'principle'
        ]
        
        # Known Pulseq functions (common ones)
        pulseq_functions = [
            'makearbitraryrf', 'maketrapezoid', 'makegausspulse',
            'makeblockpulse', 'makesincpulse', 'makeadc',
            'makedelay', 'makelabel', 'maketrigger',
            'calcduration', 'calcrfbandwidth', 'calcrfphase',
            'writeute', 'writeflash', 'writeepi', 'writespinecho',
            'writetse', 'writegradientecho', 'writebssfp'
        ]
        
        # Check if it's a direct function query
        is_function = any(
            func in query_lower 
            for func in pulseq_functions
        ) or any(
            keyword in query_lower 
            for keyword in function_keywords
        )
        
        # Check if it needs code examples
        needs_code = any(
            keyword in query_lower 
            for keyword in code_keywords
        )
        
        # Check if it's conceptual
        is_conceptual = any(
            keyword in query_lower 
            for keyword in doc_keywords
        )
        
        # Detect language preference
        language = None
        if 'matlab' in query_lower or '.m' in query_lower:
            language = 'matlab'
        elif 'python' in query_lower or '.py' in query_lower or 'pypulseq' in query_lower:
            language = 'python'
        elif 'c++' in query_lower or 'cpp' in query_lower:
            language = 'cpp'
        
        # Determine primary search type
        if is_function and not needs_code:
            primary_type = SearchType.API_FUNCTION
        elif needs_code:
            primary_type = SearchType.CODE_EXAMPLE
        elif is_conceptual:
            primary_type = SearchType.DOCUMENTATION
        else:
            primary_type = SearchType.UNIFIED
        
        # Calculate confidence
        indicators = sum([is_function, needs_code, is_conceptual])
        if indicators == 0:
            confidence = 0.3  # Low confidence, will use unified search
        elif indicators == 1:
            confidence = 0.8  # High confidence, clear intent
        else:
            confidence = 0.6  # Medium confidence, mixed intent
        
        return {
            'primary_type': primary_type,
            'search_api': is_function,
            'search_code': needs_code,
            'search_docs': is_conceptual,
            'language': language,
            'confidence': confidence
        }
    
    def search_api_functions(
        self,
        query: str,
        match_count: int = 5,
        language_filter: Optional[str] = None
    ) -> str:
        """
        Search API reference for Pulseq functions.
        
        Args:
            query: Search query
            match_count: Number of results to return
            language_filter: Optional language filter (matlab/python/cpp)
            
        Returns:
            Formatted search results
        """
        context = self.performance_monitor.start_query(query, "api_functions")
        context["language_filter"] = language_filter
        
        try:
            # Don't enhance function names - keep them as-is for better matching
            search_query = query
            
            # Create embedding for the query
            from .embeddings import create_embedding
            query_embedding = create_embedding(search_query)
            
            # Build RPC parameters
            params = {
                "query_embedding": query_embedding,
                "match_count": match_count,
                "filter": {}
            }
            
            if language_filter:
                params["language_filter"] = language_filter
            
            # Try using the match_api_functions RPC first
            try:
                result = self.supabase_client.client.rpc("match_api_functions", params).execute()
                results = result.data if result.data else []
                logger.info(f"API function search returned {len(results)} results")
            except Exception as rpc_error:
                logger.warning(f"RPC match_api_functions failed: {rpc_error}, falling back to direct query")
                results = self._fallback_api_search(query, match_count, language_filter)
            
            # Record successful completion
            self.performance_monitor.record_query_completion(context, results)
            
            # Format results
            return self._format_api_results(results, query)
            
        except Exception as e:
            # Record failure
            self.performance_monitor.record_query_completion(context, [], error=str(e))
            logger.error(f"API function search failed: {e}")
            return f"Error searching API functions: {str(e)}"
    
    def _fallback_api_search(
        self,
        query: str,
        match_count: int,
        language_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fallback API search using direct SQL query.
        """
        try:
            query_builder = (
                self.supabase_client.client.from_("api_reference")
                .select("*")
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
    
    def search_all_sources(
        self,
        query: str,
        match_count: int = 10
    ) -> str:
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
            'api_functions': [],
            'code_examples': [],
            'documentation': [],
            'strategy': strategy
        }
        
        # Determine result distribution based on classification
        if strategy['primary_type'] == SearchType.API_FUNCTION:
            # Prioritize API results
            api_count = max(1, match_count // 2)
            code_count = max(1, match_count // 4)
            doc_count = match_count - api_count - code_count
        elif strategy['primary_type'] == SearchType.CODE_EXAMPLE:
            # Prioritize code examples
            code_count = max(1, match_count // 2)
            api_count = max(1, match_count // 4)
            doc_count = match_count - api_count - code_count
        elif strategy['primary_type'] == SearchType.DOCUMENTATION:
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
            if strategy['search_api'] and api_count > 0:
                api_results = self.search_api_functions(
                    query, 
                    api_count,
                    strategy.get('language')
                )
                # Extract results from formatted string (not ideal but maintains compatibility)
                if "No API functions found" not in api_results:
                    results['api_functions'] = [{'formatted_result': api_results}]
            
            if strategy['search_code'] and code_count > 0:
                code_results = self.search_code_examples(
                    query,
                    match_count=code_count,
                    use_hybrid=True
                )
                if "No code examples found" not in code_results:
                    results['code_examples'] = [{'formatted_result': code_results}]
            
            if strategy['search_docs'] and doc_count > 0:
                doc_results = self.perform_rag_query(
                    query,
                    match_count=doc_count,
                    use_hybrid=True
                )
                if "No documentation found" not in doc_results:
                    results['documentation'] = [{'formatted_result': doc_results}]
            
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
                formatted.append(f"**Signature:**")
                formatted.append(f"```{language}")
                formatted.append(signature)
                formatted.append("```")
            
            if description:
                formatted.append(f"**Description:** {description}")
            
            if parameters and isinstance(parameters, dict):
                formatted.append(f"**Parameters:**")
                for param, details in parameters.items():
                    if isinstance(details, dict):
                        param_type = details.get('type', 'unknown')
                        param_desc = details.get('description', 'No description')
                        formatted.append(f"- `{param}` ({param_type}): {param_desc}")
                    else:
                        formatted.append(f"- `{param}`: {details}")
            
            if returns:
                formatted.append(f"**Returns:** {returns}")
            
            formatted.append("")  # Empty line between results
        
        return "\n".join(formatted)
    
    def _format_unified_results(
        self,
        results: Dict[str, List[Dict]],
        query: str
    ) -> str:
        """
        Format unified search results for display.
        """
        formatted = [f"## Search Results for: '{query}'\n"]
        
        strategy = results.get('strategy', {})
        confidence = strategy.get('confidence', 0)
        
        if confidence < 0.5:
            formatted.append("*Note: Query intent unclear, showing results from all sources*\n")
        else:
            primary_type = strategy.get('primary_type', SearchType.UNIFIED)
            formatted.append(f"*Detected intent: {primary_type.value.replace('_', ' ').title()}*\n")
        
        # Show results from each source
        sources_shown = 0
        
        # API Functions
        if results['api_functions']:
            formatted.append("### 🔧 API Functions\n")
            for result in results['api_functions']:
                formatted.append(result['formatted_result'])
            sources_shown += 1
        
        # Code Examples
        if results['code_examples']:
            if sources_shown > 0:
                formatted.append("---\n")
            formatted.append("### 💻 Code Examples\n")
            for result in results['code_examples']:
                formatted.append(result['formatted_result'])
            sources_shown += 1
        
        # Documentation
        if results['documentation']:
            if sources_shown > 0:
                formatted.append("---\n")
            formatted.append("### 📚 Documentation\n")
            for result in results['documentation']:
                formatted.append(result['formatted_result'])
            sources_shown += 1
        
        if sources_shown == 0:
            formatted.append("No results found in any source for this query.")
        
        return "\n".join(formatted)
        
    def perform_rag_query(
        self,
        query: str,
        source: Optional[str] = None,
        match_count: int = 5,
        use_hybrid: bool = True
    ) -> str:
        """
        Search the RAG database for documentation and information.
        
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
            # Perform search
            if use_hybrid and self.settings.use_hybrid_search:
                results = self.supabase_client.perform_hybrid_search(
                    query=query,
                    match_count=match_count,
                    source=source,
                    search_type="documents"
                )
            else:
                filter_metadata = {"source": source} if source else None
                results = self.supabase_client.search_documents(
                    query=query,
                    match_count=match_count,
                    filter_metadata=filter_metadata
                )
            
            # Record successful completion
            self.performance_monitor.record_query_completion(context, results)
            
            # Format results
            return self._format_rag_results(results, query)
            
        except Exception as e:
            # Record failure
            self.performance_monitor.record_query_completion(context, [], error=str(e))
            logger.error(f"RAG query failed: {e}")
            return f"Error performing RAG search: {str(e)}"
    
    def search_code_examples(
        self,
        query: str,
        source_id: Optional[str] = None,
        match_count: int = 5,
        use_hybrid: bool = True
    ) -> str:
        """
        Search for code examples and implementations.
        
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
        context["hybrid_search"] = use_hybrid
        
        try:
            # Perform search
            if use_hybrid and self.settings.use_hybrid_search:
                results = self.supabase_client.perform_hybrid_search(
                    query=query,
                    match_count=match_count,
                    source=source_id,
                    search_type="code_examples"
                )
            else:
                filter_metadata = {"source": source_id} if source_id else None
                results = self.supabase_client.search_code_examples(
                    query=query,
                    match_count=match_count,
                    filter_metadata=filter_metadata,
                    source_id=source_id
                )
            
            # Record successful completion
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
                content_preview = content[:self.doc_preview_limit] + f"\n... [Truncated - {len(content)} total characters]"
            else:
                content_preview = content
            
            # Format result
            formatted.append(f"### {i}. Result from {item.get('source_id', 'Unknown source')}")
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
        formatted.append(f"Found {len(results)} code examples:\n")
        
        for i, item in enumerate(results, 1):
            # Extract metadata
            metadata = item.get("metadata", {})
            language = metadata.get("language", "Unknown")
            
            # Get summary and full code
            summary = item.get("summary", "No description available")
            code = item.get("content", "")
            
            # Only truncate if code is extremely long
            if len(code) > self.code_preview_limit:
                code_preview = code[:self.code_preview_limit] + f"\n% ... [Truncated - {len(code)} total characters]"
            else:
                code_preview = code
            
            # Format result
            formatted.append(f"### {i}. {language.upper()} Example from {item.get('source_id', 'Unknown')}")
            formatted.append(f"**Summary:** {summary}")
            formatted.append(f"**URL:** {item.get('url', 'N/A')}")
            formatted.append(f"**Relevance:** {item.get('similarity', 0):.2%}")
            formatted.append(f"\n```{language}")
            formatted.append(code_preview)
            formatted.append("```")
            formatted.append("")  # Empty line between results
        
        return "\n".join(formatted)
    
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
            similarity_percentiles = self.performance_monitor.get_percentiles("similarity")
            query_patterns = self.performance_monitor.get_query_pattern_analysis()
            
            formatted = ["## RAG Performance Statistics\n"]
            
            if window_minutes:
                formatted.append(f"**Time Window:** Last {window_minutes} minutes\n")
            else:
                formatted.append("**Time Window:** All time\n")
            
            # Basic stats
            formatted.extend([
                f"**Total Queries:** {stats.total_queries}",
                f"**Average Duration:** {stats.avg_duration:.3f}s",
                f"**Min/Max Duration:** {stats.min_duration:.3f}s / {stats.max_duration:.3f}s",
                f"**Average Results:** {stats.avg_result_count:.1f}",
                f"**Average Similarity:** {stats.avg_similarity:.3f}",
                f"**Error Rate:** {stats.error_rate:.1%}",
                f"**Queries per Second:** {stats.queries_per_second:.2f}",
                f"**Cache Hit Rate:** {stats.cache_hit_rate:.1%}",
                ""
            ])
            
            # Duration percentiles
            if duration_percentiles:
                formatted.append("### Response Time Percentiles")
                for percentile, value in duration_percentiles.items():
                    formatted.append(f"**{percentile.upper()}:** {value:.3f}s")
                formatted.append("")
            
            # Query patterns
            if query_patterns:
                formatted.append("### Query Patterns")
                formatted.append(f"**Search Type Distribution:** {query_patterns.get('search_type_distribution', {})}")
                formatted.append(f"**Average Query Length:** {query_patterns.get('avg_query_length', 0):.1f} chars")
                formatted.append(f"**Reranking Usage:** {query_patterns.get('reranking_usage_rate', 0):.1%}")
                formatted.append(f"**Hybrid Search Usage:** {query_patterns.get('hybrid_search_usage_rate', 0):.1%}")
                formatted.append(f"**Unique Queries:** {query_patterns.get('total_unique_queries', 0)}")
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
                    formatted.append(f"{i}. **Error:** {q.error} - Query: {q.query[:50]}...")
                formatted.append("")
            
            return "\n".join(formatted)
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return f"Error retrieving performance statistics: {str(e)}"


# Global instance
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """
    Get the global RAG service instance.
    
    Returns:
        RAGService instance
    """
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service