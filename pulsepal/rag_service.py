"""
RAG (Retrieval Augmented Generation) service for Pulsepal.

This module provides the core RAG functionality for searching documentation
and code examples, with support for various search modes and result formatting.
"""

import logging
from typing import List, Dict, Any, Optional
from .supabase_client import get_supabase_client, SupabaseRAGClient
from .settings import get_settings

logger = logging.getLogger(__name__)


class RAGService:
    """Service for performing RAG queries on Pulseq documentation and code."""
    
    def __init__(self):
        """Initialize RAG service with Supabase client."""
        self._supabase_client = None
        self.settings = get_settings()
    
    @property
    def supabase_client(self) -> SupabaseRAGClient:
        """Lazy load Supabase client."""
        if self._supabase_client is None:
            self._supabase_client = get_supabase_client()
        return self._supabase_client
        
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
            
            # Format results
            return self._format_rag_results(results, query)
            
        except Exception as e:
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
            
            # Format results
            return self._format_code_results(results, query)
            
        except Exception as e:
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
            
            # Extract content preview
            content = item.get("content", "")
            content_preview = content[:500] + "..." if len(content) > 500 else content
            
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
            
            # Get summary and code preview
            summary = item.get("summary", "No description available")
            code = item.get("content", "")
            code_preview = code[:300] + "..." if len(code) > 300 else code
            
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