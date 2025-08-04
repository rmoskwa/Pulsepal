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
        # For specific sequence names, don't modify them too much
        known_sequences = ['ute', 'writeute', 'spin echo', 'gradient echo', 'epi', 'flash', 'tse', 'bssfp']
        query_lower = query.lower()
        
        # Check if query contains a known sequence name
        for seq in known_sequences:
            if seq in query_lower:
                return query  # Keep original query for specific sequences
        
        # For general queries, add context
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
            # IMPORTANT: Always provide filter as empty JSONB if None (required by RPC function)
            params = {
                "query_embedding": query_embedding,
                "match_count": search_count,
                "filter": filter_metadata if filter_metadata else {},  # Always provide filter
                "source_filter": source_id  # Can be None, SQL function handles it
            }

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
        Improved keyword search that avoids pollution from common words.
        
        Args:
            query: Search query
            match_count: Number of results per strategy
            table_name: Database table to search
            source: Optional source filter
            
        Returns:
            Combined keyword search results
        """
        results = []
        
        # Strategy 0: Exact sequence name search (for known sequences)
        known_sequences = ['writeUTE', 'writeFLASH', 'writeEPI', 'writeSpinEcho', 'writeTSE', 'writeGradientEcho', 'writebSSFP']
        
        # Special handling for UTE queries
        if 'ute' in query.lower():
            # Always search for writeUTE files when UTE is mentioned
            if table_name == "code_examples":
                ute_query = (
                    self.client.from_(table_name)
                    .select("id, url, chunk_number, content, summary, metadata, source_id")
                    .or_("url.ilike.%writeUTE%,summary.ilike.%UTE%,summary.ilike.%ultra-short echo%,summary.ilike.%ultrashort echo%")
                )
                if source:
                    ute_query = ute_query.eq("source_id", source)
                
                ute_response = ute_query.limit(match_count).execute()
                if ute_response.data:
                    logger.debug(f"Found {len(ute_response.data)} UTE-specific matches")
                    results.extend(ute_response.data)
                    # Return immediately if we found enough UTE results
                    if len(results) >= match_count:
                        return results[:match_count]
        
        
        # Also check for common variations
        sequence_found = False
        for seq_name in known_sequences:
            # Check if sequence name or common variations are in the query
            seq_lower = seq_name.lower()
            query_lower = query.lower()
            
            # Check for exact sequence name or common variations
            if seq_lower in query_lower or seq_lower.replace('write', '') in query_lower:
                logger.debug(f"Found sequence match: {seq_name} in query: {query}")
                # Search in URL and summary columns for table code_examples
                if table_name == "code_examples":
                    exact_query = (
                        self.client.from_(table_name)
                        .select("id, url, chunk_number, content, summary, metadata, source_id")
                        .or_(f"url.ilike.%{seq_name}%,summary.ilike.%{seq_name.replace('write', '')}%")
                    )
                else:
                    exact_query = (
                        self.client.from_(table_name)
                        .select("id, url, chunk_number, content, metadata, source_id")
                        .or_(f"url.ilike.%{seq_name}%,content.ilike.%{seq_name}%")
                    )
                
                if source:
                    exact_query = exact_query.eq("source_id", source)
                
                exact_response = exact_query.limit(match_count).execute()
                if exact_response.data:
                    logger.debug(f"Found {len(exact_response.data)} exact matches for {seq_name}")
                    results.extend(exact_response.data)
                    sequence_found = True
                break  # Only search for one sequence name
        
        # If we found exact matches and they're enough, return early
        if sequence_found and len(results) >= match_count:
            logger.debug(f"Returning {len(results)} exact matches early")
            return results[:match_count]
        
        # Strategy 1: Full phrase search
        # Include summary in select for code_examples table
        if table_name == "code_examples":
            select_fields = "id, url, chunk_number, content, summary, metadata, source_id"
        else:
            select_fields = "id, url, chunk_number, content, metadata, source_id"
        
        phrase_query = (
            self.client.from_(table_name)
            .select(select_fields)
            .ilike("content", f"%{query}%")
        )
        if source:
            phrase_query = phrase_query.eq("source_id", source)
        
        phrase_response = phrase_query.limit(match_count).execute()
        if phrase_response.data:
            results.extend(phrase_response.data)
        
        # Strategy 2: Individual words ONLY if they're meaningful
        words = query.split()
        # Filter out common words that pollute results
        common_words = ['sequence', 'example', 'code', 'pulse', 'make', 'with', 'from', 'function', 'method']
        meaningful_words = [w for w in words if len(w) > 3 and w.lower() not in common_words]
        
        # Only do word search if we have meaningful words left
        if meaningful_words and len(results) < match_count:
            for word in meaningful_words:
                # Include summary in select for code_examples table
                if table_name == "code_examples":
                    select_fields = "id, url, chunk_number, content, summary, metadata, source_id"
                else:
                    select_fields = "id, url, chunk_number, content, metadata, source_id"
                
                word_query = (
                    self.client.from_(table_name)
                    .select(select_fields)
                    .ilike("content", f"%{word}%")
                )
                if source:
                    word_query = word_query.eq("source_id", source)
                
                word_response = word_query.limit(match_count // len(meaningful_words)).execute()
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
        use_reranking: bool = True,
        keyword_query_override: Optional[str] = None
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
            keyword_query_override: Optional override for keyword search query
            
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
            # Use override query if provided, otherwise use original query
            keyword_query = keyword_query_override if keyword_query_override else query
            keyword_results = self._multi_strategy_keyword_search(
                keyword_query, match_count * 2, table_name, source
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