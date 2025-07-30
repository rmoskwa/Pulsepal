"""
Supabase client for Pulsepal's RAG functionality.

This module provides direct integration with Supabase for vector similarity search,
managing documentation and code examples stored in the vector database.
"""

import os
import logging
import gc
import asyncio
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
# Lazy imports to avoid slow startup
create_embedding = None
_CrossEncoder = None

def _get_create_embedding():
    """Lazy load create_embedding function."""
    global create_embedding
    if create_embedding is None:
        from .embeddings import create_embedding as _create_embedding
        create_embedding = _create_embedding
    return create_embedding

def _get_cross_encoder():
    """Lazy load CrossEncoder for reranking."""
    global _CrossEncoder
    if _CrossEncoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _CrossEncoder = CrossEncoder
        except ImportError:
            logger.warning("sentence-transformers not available for reranking")
            _CrossEncoder = None
    return _CrossEncoder

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
        self._reranker = None  # Lazy load cross-encoder
        logger.info("Supabase client initialized successfully")
    
    def _get_reranker(self, model_path: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Get or initialize the cross-encoder reranker."""
        if self._reranker is None:
            CrossEncoder = _get_cross_encoder()
            if CrossEncoder is not None:
                try:
                    self._reranker = CrossEncoder(model_path)
                    logger.info(f"Cross-encoder reranker loaded: {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load cross-encoder reranker: {e}")
                    self._reranker = None
        return self._reranker

    def rerank_results(self, query: str, results: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        """
        Rerank search results using cross-encoder for improved relevance.
        
        Args:
            query: Original search query
            results: List of search results to rerank
            top_k: Number of top results to return (None for all)
            
        Returns:
            Reranked results with updated similarity scores
        """
        if not results:
            return results
            
        try:
            reranker = self._get_reranker()
            if reranker is None:
                logger.debug("No reranker available, returning original results")
                return results[:top_k] if top_k else results
                
            # Prepare text pairs for reranking
            texts = [result.get("content", "") for result in results]
            pairs = [[query, text] for text in texts]
            
            # Get reranking scores
            scores = reranker.predict(pairs)
            
            # Add rerank scores and sort
            for i, result in enumerate(results):
                result["rerank_score"] = float(scores[i])
            
            # Sort by rerank score
            reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            logger.debug(f"Reranked {len(results)} results using cross-encoder")
            return reranked[:top_k] if top_k else reranked
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return results[:top_k] if top_k else results

    def enhance_query_for_code_search(self, query: str) -> str:
        """Enhance query for better code example matching."""
        return f"Code example for {query}\n\nImplementation showing {query}"

    def preprocess_query(self, query: str, search_type: str = "documents") -> str:
        """
        Preprocess query based on search type for better embeddings.
        
        Args:
            query: Original query
            search_type: Type of search ("documents" or "code_examples")
            
        Returns:
            Enhanced query for better embedding generation
        """
        if search_type == "code_examples":
            return self.enhance_query_for_code_search(query)
        return query

    def search_documents(
        self,
        query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        use_reranking: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents in Supabase using vector similarity.

        Args:
            query: Query text
            match_count: Maximum number of results to return
            filter_metadata: Optional metadata filter
            use_reranking: Whether to apply reranking to results

        Returns:
            List of matching documents
        """
        try:
            # Preprocess query for better embeddings
            enhanced_query = self.preprocess_query(query, "documents")
            
            # Create embedding for the query
            create_embedding_fn = _get_create_embedding()
            query_embedding = create_embedding_fn(enhanced_query)

            # Get more results for reranking if enabled
            search_count = match_count * 2 if use_reranking else match_count
            
            # Execute the search using the match_crawled_pages function
            params = {"query_embedding": query_embedding, "match_count": search_count}

            # Only add the filter if it's actually provided and not empty
            if filter_metadata:
                params["filter"] = filter_metadata

            result = self.client.rpc("match_crawled_pages", params).execute()
            
            # Apply reranking if enabled
            results = result.data if result.data else []
            if use_reranking and results:
                results = self.rerank_results(query, results, match_count)
            
            logger.info(f"Document search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def search_code_examples(
        self,
        query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        source_id: Optional[str] = None,
        use_reranking: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search for code examples in Supabase using vector similarity.

        Args:
            query: Query text
            match_count: Maximum number of results to return
            filter_metadata: Optional metadata filter
            source_id: Optional source ID to filter results
            use_reranking: Whether to apply reranking to results

        Returns:
            List of matching code examples
        """
        try:
            # Use enhanced query preprocessing
            enhanced_query = self.preprocess_query(query, "code_examples")

            # Create embedding for the enhanced query
            create_embedding_fn = _get_create_embedding()
            query_embedding = create_embedding_fn(enhanced_query)

            # Get more results for reranking if enabled
            search_count = match_count * 2 if use_reranking else match_count

            # Execute the search using the match_code_examples function
            params = {"query_embedding": query_embedding, "match_count": search_count}

            # Only add the filter if it's actually provided and not empty
            if filter_metadata:
                params["filter"] = filter_metadata

            # Add source filter if provided
            if source_id:
                params["source_filter"] = source_id

            result = self.client.rpc("match_code_examples", params).execute()
            
            # Apply reranking if enabled
            results = result.data if result.data else []
            if use_reranking and results:
                results = self.rerank_results(query, results, match_count)
            
            logger.info(f"Code search returned {len(results)} results")
            return results
            
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

    def _reciprocal_rank_fusion(
        self, 
        vector_results: List[Dict], 
        keyword_results: List[Dict], 
        match_count: int,
        k: int = 60
    ) -> List[Dict]:
        """
        Combine search results using Reciprocal Rank Fusion (RRF).
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            match_count: Number of results to return
            k: RRF parameter (typically 60)
            
        Returns:
            Fused and ranked results
        """
        # Create score maps
        vector_scores = {}
        keyword_scores = {}
        all_docs = {}
        
        # Process vector results
        for rank, doc in enumerate(vector_results):
            doc_id = doc.get("id")
            if doc_id:
                vector_scores[doc_id] = 1.0 / (k + rank + 1)
                all_docs[doc_id] = doc
        
        # Process keyword results  
        for rank, doc in enumerate(keyword_results):
            doc_id = doc.get("id")
            if doc_id:
                keyword_scores[doc_id] = 1.0 / (k + rank + 1)
                if doc_id not in all_docs:
                    # Convert keyword result to standard format
                    all_docs[doc_id] = {
                        "id": doc["id"],
                        "url": doc["url"],
                        "chunk_number": doc.get("chunk_number"),
                        "content": doc["content"],
                        "metadata": doc.get("metadata", {}),
                        "source_id": doc["source_id"],
                        "similarity": 0.5,  # Default for keyword-only
                    }
        
        # Calculate RRF scores
        rrf_scores = {}
        for doc_id in all_docs:
            rrf_scores[doc_id] = (
                vector_scores.get(doc_id, 0) + keyword_scores.get(doc_id, 0)
            )
        
        # Sort by RRF score and return top results
        sorted_docs = sorted(
            all_docs.values(),
            key=lambda x: rrf_scores.get(x["id"], 0),
            reverse=True
        )
        
        # Add RRF scores to results
        for doc in sorted_docs:
            doc["rrf_score"] = rrf_scores.get(doc["id"], 0)
        
        return sorted_docs[:match_count]

    def _multi_strategy_keyword_search(
        self, 
        query: str, 
        match_count: int, 
        table_name: str,
        source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform multiple keyword search strategies.
        
        Args:
            query: Search query
            match_count: Number of results per strategy
            table_name: Database table to search
            source: Optional source filter
            
        Returns:
            Combined keyword search results
        """
        results = []
        
        # Strategy 1: Full phrase search
        phrase_query = (
            self.client.from_(table_name)
            .select("id, url, chunk_number, content, metadata, source_id")
            .ilike("content", f"%{query}%")
        )
        if source:
            phrase_query = phrase_query.eq("source_id", source)
        
        phrase_response = phrase_query.limit(match_count).execute()
        if phrase_response.data:
            results.extend(phrase_response.data)
        
        # Strategy 2: Individual word search (if query has multiple words)
        words = query.split()
        if len(words) > 1:
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_query = (
                        self.client.from_(table_name)
                        .select("id, url, chunk_number, content, metadata, source_id")
                        .ilike("content", f"%{word}%")
                    )
                    if source:
                        word_query = word_query.eq("source_id", source)
                    
                    word_response = word_query.limit(match_count // len(words)).execute()
                    if word_response.data:
                        results.extend(word_response.data)
        
        # Remove duplicates while preserving order
        seen_ids = set()
        unique_results = []
        for result in results:
            if result["id"] not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result["id"])
        
        return unique_results[:match_count]

    def perform_hybrid_search(
        self,
        query: str,
        match_count: int = 10,
        source: Optional[str] = None,
        search_type: str = "documents",
        use_rrf: bool = True,
        use_reranking: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform advanced hybrid search combining vector and keyword search.
        
        Args:
            query: Search query
            match_count: Maximum number of results
            source: Optional source filter
            search_type: Type of search ("documents" or "code_examples")
            use_rrf: Whether to use Reciprocal Rank Fusion
            use_reranking: Whether to apply cross-encoder reranking
            
        Returns:
            Combined and ranked search results
        """
        try:
            # Vector search (disable internal reranking to do it after fusion)
            if search_type == "documents":
                vector_results = self.search_documents(
                    query, 
                    match_count * 2,
                    {"source": source} if source else None,
                    use_reranking=False  # Apply reranking after fusion
                )
                table_name = "crawled_pages"
            else:
                vector_results = self.search_code_examples(
                    query,
                    match_count * 2,
                    {"source": source} if source else None,
                    source,
                    use_reranking=False  # Apply reranking after fusion
                )
                table_name = "code_examples"
            
            # Enhanced keyword search with multiple strategies
            keyword_results = self._multi_strategy_keyword_search(
                query, match_count * 2, table_name, source
            )
            
            # Combine results using RRF or simple fusion
            if use_rrf:
                combined_results = self._reciprocal_rank_fusion(
                    vector_results, keyword_results, match_count * 2
                )
            else:
                # Fallback to original fusion method
                combined_results = self._simple_fusion(
                    vector_results, keyword_results, match_count * 2
                )
            
            # Apply final reranking if enabled
            if use_reranking and combined_results:
                combined_results = self.rerank_results(
                    query, combined_results, match_count
                )
            else:
                combined_results = combined_results[:match_count]
            
            logger.info(f"Advanced hybrid search returned {len(combined_results)} results")
            return combined_results
            
        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            return []

    def _simple_fusion(
        self, 
        vector_results: List[Dict], 
        keyword_results: List[Dict], 
        match_count: int
    ) -> List[Dict]:
        """Simple fusion method (fallback for original behavior)."""
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
                        "chunk_number": kr.get("chunk_number"),
                        "content": kr["content"],
                        "metadata": kr.get("metadata", {}),
                        "source_id": kr["source_id"],
                        "similarity": 0.5,  # Default similarity for keyword-only matches
                    }
                )
                seen_ids.add(kr["id"])
        
        return combined_results[:match_count]


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