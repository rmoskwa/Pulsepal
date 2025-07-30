"""
Supabase client for Pulsepal's RAG functionality.

This module provides direct integration with Supabase for vector similarity search,
managing documentation and code examples stored in the vector database.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
# Lazy import to avoid slow startup
create_embedding = None

def _get_create_embedding():
    """Lazy load create_embedding function."""
    global create_embedding
    if create_embedding is None:
        from .embeddings import create_embedding as _create_embedding
        create_embedding = _create_embedding
    return create_embedding

logger = logging.getLogger(__name__)


class SupabaseRAGClient:
    """Client for interacting with Supabase vector database for RAG queries."""
    
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """
        Initialize Supabase client.
        
        Args:
            url: Supabase URL (defaults to environment variable)
            key: Supabase service key (defaults to environment variable)
        """
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise ValueError(
                "Supabase URL and service key must be provided or set in environment variables "
                "(SUPABASE_URL and SUPABASE_KEY)"
            )
        
        self.client: Client = create_client(self.url, self.key)
        logger.info("Supabase client initialized successfully")
    
    def search_documents(
        self,
        query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents in Supabase using vector similarity.

        Args:
            query: Query text
            match_count: Maximum number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of matching documents
        """
        try:
            # Create embedding for the query
            create_embedding_fn = _get_create_embedding()
            query_embedding = create_embedding_fn(query)

            # Execute the search using the match_crawled_pages function
            params = {"query_embedding": query_embedding, "match_count": match_count}

            # Only add the filter if it's actually provided and not empty
            if filter_metadata:
                params["filter"] = filter_metadata

            result = self.client.rpc("match_crawled_pages", params).execute()
            
            logger.info(f"Document search returned {len(result.data)} results")
            return result.data
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def search_code_examples(
        self,
        query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        source_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for code examples in Supabase using vector similarity.

        Args:
            query: Query text
            match_count: Maximum number of results to return
            filter_metadata: Optional metadata filter
            source_id: Optional source ID to filter results

        Returns:
            List of matching code examples
        """
        try:
            # Create a more descriptive query for better embedding match
            # Since code examples are embedded with their summaries
            enhanced_query = (
                f"Code example for {query}\n\nSummary: Example code showing {query}"
            )

            # Create embedding for the enhanced query
            create_embedding_fn = _get_create_embedding()
            query_embedding = create_embedding_fn(enhanced_query)

            # Execute the search using the match_code_examples function
            params = {"query_embedding": query_embedding, "match_count": match_count}

            # Only add the filter if it's actually provided and not empty
            if filter_metadata:
                params["filter"] = filter_metadata

            # Add source filter if provided
            if source_id:
                params["source_filter"] = source_id

            result = self.client.rpc("match_code_examples", params).execute()
            
            logger.info(f"Code search returned {len(result.data)} results")
            return result.data
            
        except Exception as e:
            logger.error(f"Error searching code examples: {e}")
            return []

    def get_available_sources(self) -> List[Dict[str, Any]]:
        """
        Get all available sources from the sources table.

        Returns:
            List of available sources with their details
        """
        try:
            # Query the sources table directly
            result = (
                self.client.from_("sources")
                .select("*")
                .order("source_id")
                .execute()
            )

            # Format the sources with their details
            sources = []
            if result.data:
                for source in result.data:
                    sources.append(
                        {
                            "source_id": source.get("source_id"),
                            "summary": source.get("summary"),
                            "total_words": source.get("total_words") or source.get("total_word_count"),
                            "created_at": source.get("created_at"),
                            "updated_at": source.get("updated_at"),
                        }
                    )

            logger.info(f"Retrieved {len(sources)} sources from database")
            return sources
            
        except Exception as e:
            logger.error(f"Error retrieving sources: {e}")
            return []

    def perform_hybrid_search(
        self,
        query: str,
        match_count: int = 10,
        source: Optional[str] = None,
        search_type: str = "documents"
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and keyword search.
        
        Args:
            query: Search query
            match_count: Maximum number of results
            source: Optional source filter
            search_type: Type of search ("documents" or "code_examples")
            
        Returns:
            Combined and ranked search results
        """
        try:
            # Vector search
            if search_type == "documents":
                vector_results = self.search_documents(
                    query, 
                    match_count * 2,  # Get more for filtering
                    {"source": source} if source else None
                )
                table_name = "crawled_pages"
            else:
                vector_results = self.search_code_examples(
                    query,
                    match_count * 2,
                    {"source": source} if source else None,
                    source
                )
                table_name = "code_examples"
            
            # Keyword search using ILIKE
            keyword_query = (
                self.client.from_(table_name)
                .select("id, url, chunk_number, content, metadata, source_id")
                .ilike("content", f"%{query}%")
            )
            
            # Apply source filter if provided
            if source:
                keyword_query = keyword_query.eq("source_id", source)
            
            # Execute keyword search
            keyword_response = keyword_query.limit(match_count * 2).execute()
            keyword_results = keyword_response.data if keyword_response.data else []
            
            # Combine results with preference for items appearing in both
            seen_ids = set()
            combined_results = []
            
            # First, add items that appear in both searches (best matches)
            vector_ids = {r.get("id") for r in vector_results if r.get("id")}
            for kr in keyword_results:
                if kr["id"] in vector_ids and kr["id"] not in seen_ids:
                    # Find the vector result to get similarity score
                    for vr in vector_results:
                        if vr.get("id") == kr["id"]:
                            # Boost similarity score for items in both results
                            vr["similarity"] = min(1.0, vr.get("similarity", 0) * 1.2)
                            combined_results.append(vr)
                            seen_ids.add(kr["id"])
                            break
            
            # Then add remaining vector results
            for vr in vector_results:
                if (
                    vr.get("id")
                    and vr["id"] not in seen_ids
                    and len(combined_results) < match_count
                ):
                    combined_results.append(vr)
                    seen_ids.add(vr["id"])
            
            # Finally, add pure keyword matches if we need more results
            for kr in keyword_results:
                if kr["id"] not in seen_ids and len(combined_results) < match_count:
                    # Convert keyword result to match vector result format
                    combined_results.append(
                        {
                            "id": kr["id"],
                            "url": kr["url"],
                            "chunk_number": kr["chunk_number"],
                            "content": kr["content"],
                            "metadata": kr["metadata"],
                            "source_id": kr["source_id"],
                            "similarity": 0.5,  # Default similarity for keyword-only matches
                        }
                    )
                    seen_ids.add(kr["id"])
            
            logger.info(f"Hybrid search returned {len(combined_results)} results")
            return combined_results[:match_count]
            
        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            return []


# Global instance
_supabase_client: Optional[SupabaseRAGClient] = None


def get_supabase_client() -> SupabaseRAGClient:
    """
    Get the global Supabase client instance.
    
    Returns:
        SupabaseRAGClient instance
    """
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = SupabaseRAGClient()
    return _supabase_client