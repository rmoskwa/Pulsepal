"""
Modern RAG Service v2 - Source-aware retrieval with rich metadata and function validation.

This module provides a source-aware RAG service that intelligently routes queries
to appropriate data sources (API docs, examples, tutorials) based on user intent.
Includes deterministic function validation to prevent hallucinations.
"""

import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional

from .rag_fusion import merge_search_results, select_top_results
from .reranker_service import BGERerankerService
from .rag_formatters import format_unified_response
from .settings import get_settings
from .source_profiles import MULTI_CHUNK_DOCUMENTS
from .supabase_client import SupabaseRAGClient, get_supabase_client

logger = logging.getLogger(__name__)


class ModernPulseqRAG:
    """
    Modern RAG: Simple, fast retrieval with rich metadata.
    No pattern matching, no classification, no intelligence.
    Includes deterministic function validation for hallucination prevention.
    """

    def __init__(self):
        """Initialize RAG service with Supabase client."""
        self._supabase_client = None
        self._reranker = None
        self.settings = get_settings()

    @property
    def supabase_client(self) -> SupabaseRAGClient:
        """Lazy load Supabase client."""
        if self._supabase_client is None:
            self._supabase_client = get_supabase_client()
        return self._supabase_client

    @property
    def reranker(self) -> BGERerankerService:
        """Lazy load reranker service."""
        if self._reranker is None:
            self._reranker = BGERerankerService(
                model_path=self.settings.reranker_model_path,
                model_name=self.settings.reranker_model_name,
                batch_size=self.settings.reranker_batch_size,
                timeout=self.settings.reranker_timeout,
            )
        return self._reranker

    async def search_with_source_awareness(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        forced: bool = False,
        source_hints: Optional[Dict] = None,
        detected_functions: Optional[List[Dict]] = None,
        use_parallel: bool = True,  # New parameter to enable parallel search
    ) -> Dict:
        """
        Perform source-aware search across Supabase tables.

        Now uses hybrid search (BM25 + vector) with RRF fusion and reranking by default.

        Args:
            query: Search query (extracted by Gemini, not raw user input)
            sources: Specific sources to search (LLM-specified or None for all)
            forced: Whether this search was forced by semantic routing
            source_hints: Additional hints about which sources to prioritize
            detected_functions: Functions detected by semantic router for direct lookup
            use_parallel: Whether to use parallel BM25+vector search (default True)

        Returns:
            Formatted results organized by source with synthesis hints
        """
        # Input validation for search queries (not user messages)
        if not query or not isinstance(query, str):
            return {
                "error": "Invalid search query",
                "message": "Search query must be a non-empty string",
                "results_by_source": {},
                "search_metadata": {"error": "invalid_input"},
            }

        query = query.strip()
        if len(query) == 0:
            return {
                "error": "Empty search query",
                "message": "Search query cannot be empty",
                "results_by_source": {},
                "search_metadata": {"error": "empty_query"},
            }

        # Get settings for hybrid search configuration
        settings = get_settings()

        # Check if hybrid search is enabled
        if not settings.hybrid_search_enabled:
            logger.info("Hybrid search disabled, falling back to vector-only search")
            return await self._fallback_to_vector_search(
                query, sources, forced, source_hints, detected_functions
            )

        # Log detected functions as hints (if any)
        if detected_functions:
            func_names = [f["name"] for f in detected_functions]
            logger.info(
                f"Function hints available: {len(func_names)} functions detected as context"
            )
            logger.debug(
                f"Detected functions: {func_names[:5]}..."
            )  # Log first 5 for debugging

        try:
            # Use hybrid search as the primary approach
            logger.info("Using hybrid search with RRF fusion and reranking")

            # If no specific sources, search all tables comprehensively
            if not sources:
                # Search all 5 tables for comprehensive results
                logger.info(
                    "Auto mode: Searching all 5 tables for comprehensive results"
                )

                # Get vector results from all tables
                vector_results = await self._search_all_tables_vector(
                    query=query,
                    limit_per_table=4,  # 4 per table = up to 20 total
                    similarity_threshold=0.3,
                )

                # Get BM25 results across all tables
                bm25_results = await self._search_bm25_async(
                    query=query,
                    table_name=None,  # Search all tables
                    limit=20,
                    rank_threshold=0.01,
                )

                # Create parallel results structure for compatibility
                parallel_results = {
                    "bm25_results": bm25_results,
                    "vector_results": vector_results,
                    "metadata": {
                        "query": query,
                        "total_results": len(bm25_results) + len(vector_results),
                        "bm25_count": len(bm25_results),
                        "vector_count": len(vector_results),
                        "performance": {"search_mode": "all_tables_comprehensive"},
                    },
                }
            else:
                # Execute parallel BM25 and vector searches for specific sources
                parallel_results = await self._parallel_search(
                    query,
                    table=None,  # Will use default tables
                    limit=max(settings.hybrid_bm25_k, settings.hybrid_vector_k),
                )

            # Extract results from parallel search
            bm25_results = parallel_results.get("bm25_results", [])
            vector_results = parallel_results.get("vector_results", [])

            # Apply RRF fusion to merge results
            from .rag_fusion import merge_search_results, select_top_results

            fused_results = merge_search_results(
                bm25_results=bm25_results[: settings.hybrid_bm25_k],
                vector_results=vector_results[: settings.hybrid_vector_k],
                k=settings.hybrid_rrf_k,
            )

            # Select top results for reranking
            top_results = select_top_results(
                fused_results, top_n=settings.hybrid_rerank_top_k
            )

            # Apply neural reranking to top results
            try:
                rerank_start = time.time()
                (
                    relevance_scores,
                    reranked_results,
                ) = await self.reranker.rerank_documents(
                    query=query,
                    documents=top_results,
                    top_k=settings.hybrid_final_top_k,  # Return top 3 by default
                )
                rerank_latency = (time.time() - rerank_start) * 1000  # Convert to ms

                logger.info(
                    f"Reranking completed in {rerank_latency:.2f}ms, "
                    f"top score: {relevance_scores[0] if relevance_scores else 0:.3f}"
                )

                # Use reranked results
                final_results = reranked_results

                # Add metadata to results
                for i, result in enumerate(final_results):
                    result["_source"] = "hybrid_search"
                    result["_search_type"] = "hybrid_reranked"
                    result["_rerank_score"] = (
                        relevance_scores[i] if i < len(relevance_scores) else 0.0
                    )

                # Check if relevance scores are too low (< 0.5)
                if relevance_scores and max(relevance_scores) < 0.5:
                    logger.warning(
                        f"Low relevance scores detected (max: {max(relevance_scores):.3f}). "
                        "Consider broader search or different query."
                    )

            except Exception as e:
                # Fallback to RRF-fused results if reranking fails
                logger.warning(f"Reranking failed, using RRF-fused results: {e}")
                final_results = top_results[: settings.hybrid_final_top_k]

                # Add metadata without reranking info
                for result in final_results:
                    result["_source"] = "hybrid_search"
                    result["_search_type"] = "hybrid_rrf_only"

            # Format results for unified response
            source_results = {"hybrid_search": final_results}

            # Format results using rag_formatters
            query_context = {
                "original_query": query,
                "forced": forced,
                "source_hints": source_hints,
                "search_mode": "hybrid_reranked",
                "detected_functions": detected_functions,
                "performance": parallel_results.get("metadata", {}).get(
                    "performance", {}
                ),
            }

            # Log search summary with performance metrics
            perf = parallel_results.get("metadata", {}).get("performance", {})
            logger.info(
                f"Hybrid search complete: {len(final_results)} final results | "
                f"BM25: {len(bm25_results)} results, "
                f"Vector: {len(vector_results)} results | "
                f"Latency - BM25: {perf.get('bm25_latency_ms', 0)}ms, "
                f"Vector: {perf.get('vector_latency_ms', 0)}ms, "
                f"Total: {perf.get('total_latency_ms', 0)}ms"
            )

            return format_unified_response(source_results, query_context)

        except Exception as e:
            # Complete hybrid search failure - fallback to vector-only search
            logger.error(
                f"Hybrid search failed completely, falling back to vector search: {e}"
            )
            return await self._fallback_to_vector_search(
                query, sources, forced, source_hints, detected_functions
            )

    async def _fallback_to_vector_search(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        forced: bool = False,
        source_hints: Optional[Dict] = None,
        detected_functions: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Fallback to vector-only search when hybrid search fails or is disabled.
        This maintains the existing vector search behavior.

        Args:
            query: User's search query
            sources: Specific sources to search (LLM-specified or None for all)
            forced: Whether this search was forced by semantic routing
            source_hints: Additional hints about which sources to prioritize
            detected_functions: Functions detected by semantic router for direct lookup

        Returns:
            Formatted results using vector-only search
        """
        logger.info("Using fallback vector-only search")

        # Default to all sources with balanced top-k approach if not specified
        if not sources:
            sources = ["api_reference", "crawled_pages", "official_sequence_examples"]
            logger.info(
                f"No sources specified, using top-k from all sources: {sources}"
            )
            use_top_k = True
            top_k_per_source = 3  # Top 3 from each source = 9 total results
        else:
            logger.info(f"LLM specified sources: {sources}")
            use_top_k = False

        # Search each source with appropriate methods
        source_results = {}

        for source in sources:
            if source == "api_reference":
                # For top-k mode, limit to 3; otherwise use default 5
                limit = top_k_per_source if use_top_k else 5
                results = await self._search_api_reference(query, limit=limit)
            elif source == "crawled_pages":
                # For top-k mode, limit to 3; otherwise use default 10
                limit = top_k_per_source if use_top_k else 10
                results = await self._search_crawled_pages(query, limit=limit)
            elif source == "official_sequence_examples":
                # For top-k mode, limit to 3; otherwise use default 5
                # Note: sequence examples are large, so we keep conservative limits
                limit = top_k_per_source if use_top_k else 5
                results = await self._search_official_sequences(query, limit=limit)
            else:
                continue

            if results:
                source_results[source] = results

        # Format results using rag_formatters
        query_context = {
            "original_query": query,
            "forced": forced,
            "source_hints": source_hints,
            "search_mode": "vector_only_fallback",
            "detected_functions": detected_functions,  # Pass as hints, not directives
        }

        # Log search summary
        total_results = sum(len(results) for results in source_results.values())
        logger.info(
            f"Vector search complete: {total_results} results from {len(source_results)} sources"
        )
        if use_top_k:
            logger.info(
                f"Used balanced top-k approach: {top_k_per_source} results per source"
            )

        return format_unified_response(source_results, query_context)

    def check_function_namespace(self, function_call: str) -> Dict[str, Any]:
        """Deterministic check for function namespace correctness."""
        result = {
            "is_valid": True,
            "is_error": False,
            "correct_form": None,
            "explanation": None,
        }

        # Parse the function call
        match = re.match(
            r"(mr|seq|tra|eve|opt)((?:\.aux)?(?:\.quat)?)?\.(\w+)",
            function_call,
        )
        if not match:
            return result  # Can't parse, assume it's OK

        namespace = match.group(1)
        sub_namespace = match.group(2) or ""
        method = match.group(3)
        full_namespace = namespace + sub_namespace

        # Query function_calling_patterns view for correct usage
        try:
            query = (
                self.supabase_client.client.table("function_calling_patterns")
                .select("name, correct_usage, class_name, is_class_method")
                .eq("name", method)
            )
            response = self.supabase_client.safe_execute(query)
        except Exception as e:
            logger.warning(f"Failed to validate function namespace: {e}")
            return result  # Return original result if validation fails

        if response and response.data:
            # Check if the namespace is correct
            correct_data = response.data[0]
            correct_usage = correct_data.get("correct_usage", "")

            # Extract namespace from correct_usage pattern
            # e.g., "seq.write('filename')" -> "seq"
            # or "mr.makeTrapezoid(...)" -> "mr"
            correct_namespace = None
            if correct_usage:
                ns_match = re.match(
                    r"(mr|seq|tra|eve|opt)(?:\.aux)?(?:\.quat)?",
                    correct_usage,
                )
                if ns_match:
                    correct_namespace = ns_match.group(0)

            # If we found a correct namespace and it doesn't match
            if correct_namespace and full_namespace != correct_namespace:
                result["is_valid"] = False
                result["is_error"] = True
                result["correct_form"] = f"{correct_namespace}.{method}"
                result["explanation"] = (
                    f"{method}() belongs to the {correct_namespace} namespace, "
                    f"not {full_namespace}"
                )

        return result

    def validate_function(self, function_name: str) -> Dict[str, Any]:
        """
        Validate a Pulseq function name and provide corrections.

        Args:
            function_name: Function name to validate (e.g., 'seq.calcKspace')

        Returns:
            Dictionary with:
            - is_valid: Whether the function exists
            - correct_form: Correct function name if it's a hallucination
            - suggestions: List of similar valid functions
            - explanation: Why it's wrong and how to fix it
        """
        from .function_index import COMMON_HALLUCINATIONS, MATLAB_FUNCTIONS

        result = {
            "is_valid": True,
            "correct_form": None,
            "suggestions": [],
            "explanation": None,
        }

        # Extract just the method name if it has a namespace
        if "." in function_name:
            parts = function_name.split(".")
            method = parts[-1]
            namespace = ".".join(parts[:-1])
        else:
            method = function_name
            namespace = None

        # Check if it's a known hallucination
        if method in COMMON_HALLUCINATIONS:
            correct = COMMON_HALLUCINATIONS[method]
            if correct:
                result["is_valid"] = False
                if namespace:
                    # Check if the namespace is correct for the corrected function
                    if correct in MATLAB_FUNCTIONS.get("direct_calls", set()):
                        result["correct_form"] = f"mr.{correct}"
                    elif (
                        "Sequence" in MATLAB_FUNCTIONS.get("class_methods", {})
                        and correct in MATLAB_FUNCTIONS["class_methods"]["Sequence"]
                    ):
                        result["correct_form"] = f"seq.{correct}"
                    else:
                        result["correct_form"] = correct
                else:
                    result["correct_form"] = correct
                result["explanation"] = (
                    f"'{method}' is a common mistake. Use '{correct}' instead."
                )
            else:
                result["is_valid"] = False
                result["explanation"] = f"'{method}' does not exist in Pulseq."
            return result

        # Check if it exists in function index
        all_functions = set()
        all_functions.update(MATLAB_FUNCTIONS.get("direct_calls", set()))
        for methods in MATLAB_FUNCTIONS.get("class_methods", {}).values():
            all_functions.update(methods)

        if method not in all_functions:
            result["is_valid"] = False

            # Find similar functions
            import difflib

            similar = difflib.get_close_matches(method, all_functions, n=3, cutoff=0.6)
            if similar:
                result["suggestions"] = similar
                result["explanation"] = (
                    f"'{method}' not found. Did you mean: {', '.join(similar)}?"
                )
            else:
                result["explanation"] = f"'{method}' is not a valid Pulseq function."

        # Validate namespace if provided and function is valid
        if result["is_valid"] and namespace:
            validated_namespace = self.check_function_namespace(function_name)
            if not validated_namespace["is_valid"]:
                result["is_valid"] = False
                result["correct_form"] = validated_namespace.get("correct_form")
                result["explanation"] = validated_namespace.get("explanation")

        return result

    async def _parallel_search(
        self,
        query: str,
        table: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Execute BM25 and vector searches in parallel using asyncio.gather.

        Args:
            query: Search query text
            table: Optional specific table to search
            limit: Maximum results per search method (default 20)

        Returns:
            Dictionary with results from both search methods and performance metrics
        """
        # Performance timing
        start_time = time.perf_counter()
        bm25_start = vector_start = 0
        bm25_end = vector_end = 0

        # Create async tasks for parallel execution
        async def timed_bm25_search():
            nonlocal bm25_start, bm25_end
            bm25_start = time.perf_counter()
            try:
                results = await self._search_bm25_async(
                    query=query, table_name=table, limit=limit
                )
                bm25_end = time.perf_counter()
                return results
            except Exception as e:
                bm25_end = time.perf_counter()
                logger.error(
                    "BM25 search failed in parallel execution",
                    extra={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "query": query[:100] if query else None,
                    },
                )
                return []

        async def timed_vector_search():
            nonlocal vector_start, vector_end
            vector_start = time.perf_counter()
            try:
                results = await self._search_vector_async(
                    query=query, table=table, limit=limit
                )
                vector_end = time.perf_counter()
                return results
            except Exception as e:
                vector_end = time.perf_counter()
                logger.error(
                    "Vector search failed in parallel execution",
                    extra={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "query": query[:100] if query else None,
                    },
                )
                return []

        # Execute both searches in parallel with error handling
        bm25_results, vector_results = await asyncio.gather(
            timed_bm25_search(),
            timed_vector_search(),
            return_exceptions=False,  # We handle exceptions inside the tasks
        )

        # Calculate latencies
        total_time = time.perf_counter() - start_time
        bm25_latency = bm25_end - bm25_start if bm25_end else 0
        vector_latency = vector_end - vector_start if vector_end else 0

        # Log performance metrics at INFO level
        logger.info(
            f"Parallel search completed for query: '{query[:50]}...' | "
            f"Total: {total_time:.3f}s | "
            f"BM25: {bm25_latency:.3f}s ({len(bm25_results)} results) | "
            f"Vector: {vector_latency:.3f}s ({len(vector_results)} results)"
        )

        # Combine results with metadata
        combined_results = {
            "bm25_results": bm25_results,
            "vector_results": vector_results,
            "metadata": {
                "query": query,
                "total_results": len(bm25_results) + len(vector_results),
                "bm25_count": len(bm25_results),
                "vector_count": len(vector_results),
                "performance": {
                    "total_latency_ms": round(total_time * 1000, 2),
                    "bm25_latency_ms": round(bm25_latency * 1000, 2),
                    "vector_latency_ms": round(vector_latency * 1000, 2),
                    "parallel_efficiency": round(
                        max(bm25_latency, vector_latency) / total_time * 100, 1
                    )
                    if total_time > 0
                    else 0,  # How close to perfect parallelization
                },
                "partial_failure": (len(bm25_results) == 0)
                != (len(vector_results) == 0),
            },
        }

        # Maintain source tracking for each result
        for result in combined_results["bm25_results"]:
            if "_source" not in result:
                result["_source"] = "bm25_search"
            if "score" not in result:
                result["score"] = result.get("rank", 0)

        for result in combined_results["vector_results"]:
            if "_source" not in result:
                result["_source"] = "vector_search"
            if "score" not in result:
                result["score"] = result.get("similarity", 0)

        return combined_results

    async def _search_vector_async(
        self,
        query: str,
        table: Optional[str] = None,
        limit: int = 20,
        similarity_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Perform async vector similarity search with proper async pattern.

        Args:
            query: Query text for vector search
            table: Optional specific table to search (if None, searches default tables)
            limit: Maximum number of results (default 20)
            similarity_threshold: Minimum similarity threshold

        Returns:
            List of vector search results with source attribution
        """
        # If a specific table is requested, use the single table search
        if table:
            return await self._search_vector_single_table(
                query=query, table=table, limit=limit
            )

        # Otherwise, search the default tables (crawled_pages and api_reference)
        results = []

        try:
            # Get embedding for query
            from .embeddings import create_embedding

            # Run embedding generation in executor for async
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(None, create_embedding, query)

            # Execute vector searches in parallel with retry for Windows connection issues
            async def search_crawled_pages():
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        return await loop.run_in_executor(
                            None,
                            lambda: self.supabase_client.rpc(
                                "match_crawled_pages",
                                {
                                    "query_embedding": query_embedding,
                                    "match_threshold": similarity_threshold,
                                    "match_count": limit // 2,  # Split between sources
                                },
                            ).execute(),
                        )
                    except OSError as e:
                        if "[WinError 10035]" in str(e) and attempt < max_retries - 1:
                            await asyncio.sleep(
                                0.1 * (attempt + 1)
                            )  # Exponential backoff
                            continue
                        raise

            async def search_api_reference():
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        return await loop.run_in_executor(
                            None,
                            lambda: self.supabase_client.rpc(
                                "match_api_reference",
                                {
                                    "query_embedding": query_embedding,
                                    "match_threshold": 0.5,  # Higher threshold for API reference
                                    "match_count": limit // 2,
                                },
                            ).execute(),
                        )
                    except OSError as e:
                        if "[WinError 10035]" in str(e) and attempt < max_retries - 1:
                            await asyncio.sleep(
                                0.1 * (attempt + 1)
                            )  # Exponential backoff
                            continue
                        raise

            # Execute both searches in parallel
            crawled_response, api_response = await asyncio.gather(
                search_crawled_pages(),
                search_api_reference(),
                return_exceptions=True,  # Don't fail if one search fails
            )

            # Process crawled pages results
            if not isinstance(crawled_response, Exception):
                if hasattr(crawled_response, "data") and crawled_response.data:
                    for item in crawled_response.data:
                        doc = {
                            "content": item.get("content", ""),
                            "url": item.get("url", ""),
                            "title": item.get("title", ""),
                            "similarity": item.get("similarity", 0),
                            "_source": "crawled_pages",
                            "_search_type": "vector",
                            "_type": self._detect_content_type(item.get("url", "")),
                            "source_attribution": "Vector search from crawled_pages",
                        }
                        results.append(doc)
            else:
                logger.error(f"Crawled pages vector search failed: {crawled_response}")

            # Process API reference results
            if not isinstance(api_response, Exception):
                if hasattr(api_response, "data") and api_response.data:
                    for item in api_response.data:
                        doc = {
                            "content": item.get("content", ""),
                            "function_name": item.get("function_name", ""),
                            "category": item.get("category", ""),
                            "similarity": item.get("similarity", 0),
                            "_source": "api_reference",
                            "_search_type": "vector",
                            "_type": "api_function",
                            "source_attribution": "Vector search from api_reference",
                        }
                        results.append(doc)
            else:
                logger.error(f"API reference vector search failed: {api_response}")

            return results

        except Exception as e:
            logger.error(
                "Async vector search failed",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "query": query[:100] if query else None,
                    "limit": limit,
                },
            )
            return []

    async def _search_all_tables_vector(
        self,
        query: str,
        limit_per_table: int = 3,
        similarity_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Search all 5 tables in parallel for comprehensive results.

        Args:
            query: Query text for vector search
            limit_per_table: Maximum results per table (default 3)
            similarity_threshold: Minimum similarity threshold

        Returns:
            Combined results from all tables with source attribution
        """
        try:
            # Get embedding for query
            from .embeddings import create_embedding

            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(None, create_embedding, query)

            # Define all tables and their RPC functions
            table_configs = [
                (
                    "pulseq_sequences",
                    "match_pulseq_sequences",
                    0.4,
                ),  # Lower threshold for sequences
                (
                    "api_reference",
                    "match_api_reference",
                    0.5,
                ),  # Higher threshold for API
                ("sequence_chunks", "match_sequence_chunks", 0.3),
                ("crawled_code", "match_crawled_code", 0.3),
                ("crawled_docs", "match_crawled_docs", 0.3),
            ]

            # Create semaphore to limit concurrent searches (max 3 at a time)
            semaphore = asyncio.Semaphore(3)

            # Create search tasks for all tables with resource limiting
            async def search_table(table_name: str, rpc_func: str, threshold: float):
                async with semaphore:  # Limit concurrent searches
                    try:
                        response = await loop.run_in_executor(
                            None,
                            lambda: self.supabase_client.client.rpc(
                                rpc_func,
                                {
                                    "query_embedding": query_embedding,
                                    "match_threshold": threshold,
                                    "match_count": limit_per_table,
                                },
                            ).execute(),
                        )

                        results = []
                        if hasattr(response, "data") and response.data:
                            for item in response.data:
                                # Add common fields
                                item["_source"] = table_name
                                item["_search_type"] = "vector"
                                item["source_table"] = table_name
                                results.append(item)

                        return results

                    except Exception as e:
                        logger.warning(f"Search failed for {table_name}: {e}")
                        return []

            # Execute all searches in parallel with resource limits
            tasks = [
                search_table(name, rpc, thresh) for name, rpc, thresh in table_configs
            ]
            all_results = await asyncio.gather(*tasks)

            # Flatten results
            combined_results = []
            for table_results in all_results:
                combined_results.extend(table_results)

            # Sort by similarity score
            combined_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)

            return combined_results

        except Exception as e:
            logger.error(f"All-tables vector search failed: {e}")
            return []

    async def _search_api_reference(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search API reference with higher precision threshold.
        Focuses on function signatures, parameters, and API specifications.
        """
        results = []

        # Get embedding for query
        from .embeddings import create_embedding

        query_embedding = create_embedding(query)

        # Search using the appropriate RPC function for API reference
        try:
            # Try the main API reference search
            api_response = self.supabase_client.rpc(
                "match_api_reference",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.5,  # Balanced threshold to catch semantic matches
                    "match_count": limit,
                },
            ).execute()

            if hasattr(api_response, "data") and api_response.data:
                for item in api_response.data:
                    results.append(
                        {
                            "function_name": item.get(
                                "name", ""
                            ),  # Fixed: DB returns 'name' not 'function_name'
                            "description": item.get("description", ""),
                            "parameters": item.get("parameters"),
                            "returns": item.get("returns"),
                            "signature": item.get("signature", ""),
                            "language": item.get("language", "matlab"),
                            "correct_usage": item.get("correct_usage", ""),
                            "category": item.get("category", ""),
                            "similarity": item.get("similarity", 0),
                            "_source": "api_reference",
                        },
                    )
        except Exception as e:
            logger.warning(f"API reference search failed: {e}")

        return results

    async def _search_crawled_pages(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search crawled pages with chunk coherence.
        Handles multi-chunk documents by retrieving all related chunks.
        """
        results = []

        # Get embedding for query
        from .embeddings import create_embedding

        query_embedding = create_embedding(query)

        # Search crawled pages
        crawled_response = self.supabase_client.rpc(
            "match_crawled_pages",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.3,  # Lower threshold for crawled pages
                "match_count": limit,
            },
        ).execute()

        if hasattr(crawled_response, "data") and crawled_response.data:
            # Check for multi-chunk documents
            urls_to_check = set()

            for item in crawled_response.data:
                url = item.get("url", "")

                # Check if this is a known multi-chunk document
                for doc_name in MULTI_CHUNK_DOCUMENTS:
                    if doc_name in url:
                        urls_to_check.add(url)
                        break

                # Add the result
                results.append(
                    {
                        "content": item.get("content", ""),
                        "url": url,
                        "title": item.get("title", ""),
                        "chunk_number": item.get("chunk_number", 0),
                        "similarity": item.get("similarity", 0),
                        "metadata": item.get("metadata", {}),
                        "_source": "crawled_pages",
                    },
                )

            # Retrieve additional chunks for multi-chunk documents
            for url in urls_to_check:
                additional_chunks = await self._retrieve_all_chunks(url)
                # Add chunks that aren't already in results
                existing_chunks = {
                    (r["url"], r.get("chunk_number", 0)) for r in results
                }
                for chunk in additional_chunks:
                    chunk_id = (chunk["url"], chunk.get("chunk_number", 0))
                    if chunk_id not in existing_chunks:
                        results.append(chunk)

        return results

    async def _search_official_sequences(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict]:
        """
        Search official sequences for tutorials.
        Returns complete educational sequences with summaries.
        Limited to 5 results to prevent overwhelming Gemini (5 sequences ≈ 75KB).
        Even for exploration queries, 5 diverse examples are sufficient.
        """
        results = []

        # Get embedding for query
        from .embeddings import create_embedding

        query_embedding = create_embedding(query)

        # Search official sequence examples using vector similarity
        try:
            # Use the RPC function for official sequences with embeddings
            sequence_response = self.supabase_client.rpc(
                "match_official_sequences",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.4,  # Lower threshold for better discovery of all example types
                    "match_count": limit,
                },
            ).execute()

            if hasattr(sequence_response, "data") and sequence_response.data:
                for item in sequence_response.data:
                    results.append(
                        {
                            "content": item.get("content", ""),
                            "file_name": item.get("file_name", ""),
                            "sequence_type": item.get("sequence_type", ""),
                            "trajectory_type": item.get("trajectory_type", ""),
                            "acceleration": item.get("acceleration", ""),
                            "ai_summary": item.get("ai_summary", ""),
                            "similarity": item.get("similarity", 0),
                            "url": item.get("url", ""),
                            "_source": "official_sequence_examples",
                        },
                    )

        except Exception as e:
            logger.warning(f"Official sequence search failed: {e}")

        return results

    def _preprocess_bm25_query(self, query: str) -> str:
        """
        Preprocess search query for BM25 database search.

        IMPORTANT: This receives search queries extracted by Gemini, NOT raw user input.
        User input (including code) is handled by Gemini before search queries are generated.

        The improved search_bm25_improved function in the database now handles:
        - Lowercase conversion
        - OR logic for better recall
        - Stop word handling

        We handle query sanitization for database safety here.

        Args:
            query: Search query string (already extracted/generated by Gemini)

        Returns:
            Sanitized and preprocessed query string
        """
        # Input validation
        if not query or not isinstance(query, str):
            return ""

        # Safety truncation for edge cases
        if len(query) > 1000:
            query = query[:1000]
            # Try to preserve complete words at the boundary
            last_space = query.rfind(" ")
            if last_space > 950:
                query = query[:last_space]
            logger.debug(f"Truncated query to {len(query)} chars")

        # Remove potentially dangerous SQL characters while preserving search operators
        # Keep: alphanumeric, spaces, hyphens, plus, parentheses, quotes, ampersand, pipe, dots
        # Remove: semicolons, backslashes, null bytes, etc.
        import re

        sanitized = re.sub(r'[^\w\s\-\+\(\)"\'\'&|\.]', " ", query)

        # Normalize whitespace
        processed = " ".join(sanitized.split())

        # Ensure non-empty result
        if not processed.strip():
            processed = "*"  # Wildcard search if everything was stripped

        return processed

    async def _search_bm25_async(
        self,
        query: str,
        table_name: Optional[str] = None,
        limit: int = 20,
        rank_threshold: float = 0.01,
    ) -> List[Dict]:
        """
        Perform async BM25 full-text search on keyword_for_search columns.

        Args:
            query: Query text for BM25 search
            table_name: Optional specific table to search (None for all tables)
            limit: Maximum number of results (default 20)
            rank_threshold: Minimum rank score threshold (default 0.01)

        Returns:
            List of BM25 search results with source attribution
        """
        try:
            # Run the synchronous BM25 search in an executor for true async
            loop = asyncio.get_event_loop()

            # Preprocess query following BM25 best practices
            processed_query = self._preprocess_bm25_query(query)

            # Search with processed query
            # The improved search_bm25_improved function now handles OR logic
            # and fallback mechanisms in the database
            results = await loop.run_in_executor(
                None,
                self.supabase_client.search_bm25,
                processed_query,
                table_name,
                limit,
                rank_threshold,
            )

            # Add source tracking and formatting to results
            for result in results:
                result["_source"] = "bm25_search"
                result["_search_type"] = "keyword"
                # Ensure source attribution
                if not result.get("source_attribution"):
                    result["source_attribution"] = (
                        f"BM25 search from {result.get('source_table', 'unknown')}"
                    )

            return results

        except Exception as e:
            logger.error(f"Async BM25 search failed: {e}")
            # Return empty list on failure for graceful degradation
            return []

    async def _search_bm25(
        self,
        query: str,
        table_name: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """
        Perform BM25 full-text search on keyword_for_search columns.
        Legacy method kept for backward compatibility.

        Args:
            query: Query text for BM25 search
            table_name: Optional specific table to search (None for all tables)
            limit: Maximum number of results

        Returns:
            List of BM25 search results
        """
        # Use the new async method
        return await self._search_bm25_async(query, table_name, limit)

    async def _search_hybrid(
        self,
        query: str,
        table: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """
        Perform hybrid search using parallel BM25 and vector searches with RRF fusion.
        Uses Reciprocal Rank Fusion for improved result merging.

        Args:
            query: Query text
            table: Optional specific table to search
            limit: Maximum number of results

        Returns:
            List of combined and re-ranked results using RRF
        """
        try:
            # Use the new parallel search implementation
            parallel_results = await self._parallel_search(
                query, table=table, limit=limit * 2
            )

            # Extract results from parallel search
            vector_results = parallel_results.get("vector_results", [])
            bm25_results = parallel_results.get("bm25_results", [])

            # Log performance from parallel search
            perf = parallel_results.get("metadata", {}).get("performance", {})
            if perf:
                logger.debug(
                    f"Hybrid search using parallel execution: "
                    f"BM25: {perf.get('bm25_latency_ms', 0)}ms, "
                    f"Vector: {perf.get('vector_latency_ms', 0)}ms, "
                    f"Efficiency: {perf.get('parallel_efficiency', 0)}%"
                )

            # Get RRF k parameter from settings
            settings = get_settings()
            k = settings.hybrid_rrf_k

            # Apply RRF fusion to merge results
            fused_results = merge_search_results(
                bm25_results=bm25_results, vector_results=vector_results, k=k
            )

            # Select top results for reranking
            top_results = select_top_results(fused_results, top_n=15)

            # Apply neural reranking to top 15 results
            try:
                rerank_start = time.time()
                (
                    relevance_scores,
                    reranked_results,
                ) = await self.reranker.rerank_documents(
                    query=query,
                    documents=top_results,
                    top_k=3,  # Return top 3 reranked results to LLM
                )
                rerank_latency = (time.time() - rerank_start) * 1000  # Convert to ms

                logger.info(
                    f"Reranking completed in {rerank_latency:.2f}ms, "
                    f"top score: {relevance_scores[0] if relevance_scores else 0:.3f}"
                )

                # Use reranked results if successful
                sorted_results = reranked_results[:limit]

                # Add reranking metadata
                for i, result in enumerate(sorted_results):
                    result["_source"] = "hybrid_search"
                    result["_search_type"] = "hybrid"
                    result["_parallel_search"] = True
                    result["_reranked"] = True
                    result["_rerank_score"] = (
                        relevance_scores[i] if i < len(relevance_scores) else 0.0
                    )

            except Exception as e:
                # Fallback to RRF-only results if reranking fails
                logger.warning(f"Reranking failed, using RRF-only results: {e}")
                sorted_results = top_results[:limit]

                # Add metadata without reranking info
                for result in sorted_results:
                    result["_source"] = "hybrid_search"
                    result["_search_type"] = "hybrid"
                    result["_parallel_search"] = True
                    result["_reranked"] = False

            logger.info(
                f"Hybrid search returned {len(sorted_results)} results "
                f"(from {len(bm25_results)} BM25 + {len(vector_results)} vector)"
            )
            return sorted_results

        except Exception as e:
            logger.error(
                "Hybrid search failed",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "query": query[:100] if query else None,
                    "table": table,
                    "limit": limit,
                },
            )
            return []

    async def _direct_function_lookup(
        self, detected_functions: List[Dict]
    ) -> List[Dict]:
        """
        Direct database lookup for detected functions.
        Returns comprehensive documentation for each function.

        Args:
            detected_functions: List of detected function info from semantic router

        Returns:
            List of complete function documentation from api_reference
        """
        results = []

        for func_info in detected_functions:
            func_name = func_info["name"]

            try:
                # Query specific fields (not SELECT * to avoid unnecessary data)
                response = (
                    self.supabase_client.client.table("api_reference")
                    .select(
                        "name, signature, description, "
                        "parameters, returns, usage_examples, "
                        "function_type, class_name, is_class_method, "
                        "calling_pattern, related_functions, "
                        "search_terms, pulseq_version",
                    )
                    .ilike("name", func_name)
                    .eq("language", "matlab")
                    .execute()
                )

                if response.data:
                    for item in response.data:
                        # Add detection metadata
                        item["_detection"] = {
                            "confidence": func_info["confidence"],
                            "type": func_info["type"],
                            "namespace": func_info.get("namespace"),
                            "original_query": func_info.get("full_match"),
                        }
                        item["_source"] = "direct_function_lookup"
                        results.append(item)

                        logger.info(f"Direct lookup found: {func_name}")
                else:
                    logger.warning(f"Direct lookup found no results for: {func_name}")

            except Exception as e:
                logger.error(f"Direct lookup failed for {func_name}: {e}")
                # Continue with other functions even if one fails

        return results

    async def _retrieve_all_chunks(self, url: str) -> List[Dict]:
        """
        Retrieve all chunks for a multi-chunk document.
        Critical for documents like specification.pdf (9 chunks).
        """
        chunks = []

        try:
            # Query all chunks with the same URL
            response = (
                self.supabase_client.client.table("crawled_pages")
                .select("*")
                .eq("url", url)
                .execute()
            )

            if response.data:
                for item in response.data:
                    chunks.append(
                        {
                            "content": item.get("content", ""),
                            "url": item.get("url", ""),
                            "title": item.get("title", ""),
                            "chunk_number": item.get("chunk_number", 0),
                            "metadata": item.get("metadata", {}),
                            "_source": "crawled_pages",
                        },
                    )

                # Sort by chunk number
                chunks.sort(key=lambda x: x.get("chunk_number", 0))

        except Exception as e:
            logger.warning(f"Failed to retrieve all chunks for {url}: {e}")

        return chunks

    def _detect_content_type(self, url: str) -> str:
        """Detect content type from URL extension only."""
        if not url:
            return "unknown"
        url_lower = url.lower()
        # Check path patterns first
        if "/examples/" in url_lower or "/demo" in url_lower:
            return "example"
        if "/api/" in url_lower or "/reference/" in url_lower:
            return "api_reference"
        # Then check extensions
        if url_lower.endswith(".m"):
            return "matlab_code"
        if url_lower.endswith(".py"):
            return "python_code"
        if url_lower.endswith(".md"):
            return "markdown_doc"
        if url_lower.endswith(".html"):
            return "html_doc"
        return "documentation"

    async def search_table(
        self,
        table: str,
        query: str,
        limit: int = 10,
        detected_functions: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Search a specific table with standardized return format.

        Args:
            table: Table name to search
            query: Search query
            limit: Maximum results
            detected_functions: Optional function hints

        Returns:
            Standardized JSON response with relationships
        """
        start_time = time.time()

        # Input validation
        if not query or not isinstance(query, str):
            return {
                "error": "Invalid query",
                "message": "Query must be a non-empty string",
                "suggestion": "Please provide a valid search query.",
            }

        query = query.strip()
        if len(query) == 0:
            return {
                "error": "Empty query",
                "message": "Query cannot be empty or only whitespace",
                "suggestion": "Please provide a valid search query.",
            }

        # Search queries (not user messages) should be reasonably short
        # If Gemini sends a very long search query, something is wrong
        if len(query) > 1000:
            query = query[:1000]
            logger.debug("Truncated query to 1000 chars")

        # Validate limit
        if not isinstance(limit, int) or limit < 1:
            limit = 10
        elif limit > 100:
            limit = 100  # Cap at reasonable maximum

        # Validate table name
        valid_tables = [
            "api_reference",
            "pulseq_sequences",
            "sequence_chunks",
            "crawled_code",
            "crawled_docs",
        ]
        if table not in valid_tables:
            return {
                "error": "Invalid table name",
                "message": f"Table '{table}' does not exist",
                "available_tables": valid_tables,
                "suggestion": "Please specify one of the available tables or use 'auto' for automatic selection.",
            }

        try:
            # Use hybrid search for the specific table
            if self.settings.hybrid_search_enabled:
                results = await self._search_hybrid(
                    query=query, table=table, limit=limit
                )
            else:
                # Fallback to vector search
                results = await self._search_vector_single_table(
                    query=query, table=table, limit=limit
                )

            # Format results based on table type
            formatted_response = self._format_table_response(
                table=table,
                results=results,
                query=query,
                execution_time=int((time.time() - start_time) * 1000),
            )

            return formatted_response

        except Exception as e:
            logger.error(f"Error searching table {table}: {e}")
            return {
                "error": "Search failed",
                "message": str(e),
                "table": table,
                "query": query,
            }

    async def search_by_ids(self, table: str, ids: List[str]) -> Dict:
        """
        Search for specific records by ID.

        Args:
            table: Table to search
            ids: List of IDs to retrieve

        Returns:
            Records matching the IDs with relationships
        """
        start_time = time.time()

        try:
            # Convert IDs to integers and filter out invalid ones
            int_ids = []
            for id_str in ids:
                try:
                    int_ids.append(int(id_str))
                except ValueError:
                    logger.warning(f"Invalid ID format: {id_str}")

            if not int_ids:
                return {
                    "source_table": table,
                    "query": f"ID:{','.join(ids)}",
                    "total_results": 0,
                    "results": [],
                    "search_metadata": {
                        "search_type": "id_lookup",
                        "message": "No valid IDs provided",
                        "suggestion": "Ensure IDs are numeric values",
                    },
                }

            # Single query using IN clause for efficiency
            response = (
                self.supabase_client.client.table(table)
                .select("*")
                .in_("id", int_ids)
                .execute()
            )
            results = response.data if response.data else []

            if not results:
                return {
                    "source_table": table,
                    "query": f"ID:{','.join(ids)}",
                    "total_results": 0,
                    "results": [],
                    "search_metadata": {
                        "search_type": "id_lookup",
                        "message": f"No records found with IDs {ids}",
                        "suggestion": "The referenced IDs may have been removed or updated. Try a text-based search instead.",
                    },
                }

            # Format with relationships
            formatted_response = self._format_table_response(
                table=table,
                results=results,
                query=f"ID:{','.join(ids)}",
                execution_time=int((time.time() - start_time) * 1000),
                search_type="id_lookup",
            )

            return formatted_response

        except Exception as e:
            logger.error(f"Error searching by IDs in {table}: {e}")
            return {
                "error": "ID lookup failed",
                "message": str(e),
                "table": table,
                "ids": ids,
            }

    async def search_by_filename(self, table: str, filename: str) -> Dict:
        """
        Search for records by filename.

        Args:
            table: Table to search
            filename: Filename to search for

        Returns:
            Records matching the filename
        """
        start_time = time.time()

        try:
            # Determine the filename column based on table
            filename_columns = {
                "pulseq_sequences": "file_name",
                "crawled_code": "file_name",
                "crawled_docs": "resource_uri",
            }

            if table not in filename_columns:
                return {
                    "error": "Filename search not supported",
                    "message": f"Table '{table}' does not support filename search",
                    "suggestion": "Use text-based search instead",
                }

            column = filename_columns[table]

            # Support wildcards
            if "*" in filename:
                # Use ILIKE for pattern matching
                pattern = filename.replace("*", "%")
                response = (
                    self.supabase_client.client.table(table)
                    .select("*")
                    .ilike(column, pattern)
                    .execute()
                )
            else:
                # Exact match
                response = (
                    self.supabase_client.client.table(table)
                    .select("*")
                    .eq(column, filename)
                    .execute()
                )

            results = response.data if response.data else []

            # Format response
            formatted_response = self._format_table_response(
                table=table,
                results=results,
                query=f"FILE:{filename}",
                execution_time=int((time.time() - start_time) * 1000),
                search_type="filename_lookup",
            )

            return formatted_response

        except Exception as e:
            logger.error(f"Error searching by filename in {table}: {e}")
            return {
                "error": "Filename search failed",
                "message": str(e),
                "table": table,
                "filename": filename,
            }

    async def _search_vector_single_table(
        self, query: str, table: str, limit: int = 10
    ) -> List[Dict]:
        """Vector search on a single table."""
        try:
            # Use existing embedding pattern
            from .embeddings import create_embedding

            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, create_embedding, query)

            # Map table names to RPC functions (based on actual database functions)
            rpc_functions = {
                "api_reference": "match_api_reference",
                "pulseq_sequences": "match_pulseq_sequences",
                "sequence_chunks": "match_sequence_chunks",
                "crawled_code": "match_crawled_code",
                "crawled_docs": "match_crawled_docs",
                "crawled_pages": "match_crawled_pages",
            }

            if table in rpc_functions:
                # Use existing RPC functions for other tables
                response = await loop.run_in_executor(
                    None,
                    lambda: self.supabase_client.client.rpc(
                        rpc_functions[table],
                        {
                            "query_embedding": embedding,
                            "match_threshold": 0.3,
                            "match_count": limit,
                        },
                    ).execute(),
                )
            else:
                logger.error(f"No search method available for table {table}")
                return []

            return response.data if response.data else []

        except Exception as e:
            logger.error(f"Vector search failed for table {table}: {e}")
            return []

    def _get_table_formatting_strategy(self, table: str) -> Dict[str, Any]:
        """
        Get the formatting strategy for a specific table.

        Returns a dict with formatter method name and hint generation logic.
        """
        strategies = {
            "api_reference": {
                "formatter": self._format_api_reference_results,
                "hint": lambda r: "CRITICAL: Use this information when generating Pulseq code to prevent hallucinations",
                "relationships": lambda r: {},
            },
            "pulseq_sequences": {
                "formatter": self._format_pulseq_sequences_results,
                "hint": lambda r: self._generate_sequences_hint(r),
                "relationships": lambda r: {"local_dependencies": True}
                if r and any(x.get("dependencies") for x in r)
                else {},
            },
            "sequence_chunks": {
                "formatter": self._format_sequence_chunks_results,
                "hint": lambda r: self._generate_chunks_hint(r),
                "relationships": lambda r: {"parent_sequence": True} if r else {},
            },
            "crawled_code": {
                "formatter": self._format_crawled_code_results,
                "hint": lambda r: self._generate_crawled_code_hint(r),
                "relationships": lambda r: {"parent_sequences": True}
                if r and any(x.get("parent_sequences") for x in r)
                else {},
            },
            "crawled_docs": {
                "formatter": self._format_crawled_docs_results,
                "hint": lambda r: self._generate_docs_hint(r),
                "relationships": lambda r: {
                    "parent_sequences": any(x.get("parent_sequences") for x in r)
                }
                if r
                else {},
            },
        }
        return strategies.get(
            table,
            {
                "formatter": lambda x: x,  # Default: return raw results
                "hint": lambda r: "",
                "relationships": lambda r: {},
            },
        )

    def _generate_crawled_code_hint(self, results: List[Dict]) -> str:
        """Generate specific hint for crawled_code results with IDs."""
        if not results:
            return ""

        parent_ids = set()
        for r in results:
            if r.get("parent_sequences"):
                parent_ids.update(r["parent_sequences"])

        if parent_ids:
            id_list = sorted(list(parent_ids))[:5]  # Show up to 5 IDs
            return f"This helper is used by {len(parent_ids)} sequences. Search 'pulseq_sequences' with IDs {id_list} to see usage examples."
        return ""

    def _generate_sequences_hint(self, results: List[Dict]) -> str:
        """Generate specific hint for pulseq_sequences results with dependency counts."""
        if not results:
            return ""

        total_deps = 0
        sequence_ids = []
        for r in results:
            if r.get("id"):
                sequence_ids.append(r["id"])
            deps = r.get("dependencies", {})
            if isinstance(deps, dict):
                local_deps = deps.get("local_dependencies", deps.get("local", []))
                if local_deps:
                    total_deps += len(local_deps)

        if total_deps > 0:
            return f"Found {len(results)} sequences with {total_deps} local helper functions. Search 'crawled_code' WHERE parent_sequences contains {sequence_ids[0] if sequence_ids else 'sequence_id'} to retrieve their implementations."
        elif len(results) > 0:
            return f"Found {len(results)} sequences. Use their IDs to search for related chunks, code, or documentation."
        return ""

    def _generate_chunks_hint(self, results: List[Dict]) -> str:
        """Generate specific hint for sequence_chunks results with parent IDs."""
        if not results:
            return ""

        # Collect unique sequence IDs
        sequence_ids = set()
        for r in results:
            if r.get("sequence_id"):
                sequence_ids.add(r["sequence_id"])

        if sequence_ids:
            id_list = sorted(list(sequence_ids))[:3]  # Show up to 3 parent IDs
            chunk_count = len(results)
            parent_count = len(sequence_ids)

            if parent_count == 1:
                return f"Found {chunk_count} chunks from sequence ID {id_list[0]}. Search 'pulseq_sequences' with ID:{id_list[0]} to retrieve the full sequence."
            else:
                return f"Found {chunk_count} chunks from {parent_count} different sequences. Use sequence_id values {id_list} to retrieve full sequences from pulseq_sequences table."
        return f"Found {len(results)} chunks. Check sequence_id field to retrieve parent sequences."

    def _generate_docs_hint(self, results: List[Dict]) -> str:
        """Generate specific hint for crawled_docs results."""
        if not results:
            return ""

        # Check for parent sequences
        parent_ids = set()
        for r in results:
            if r.get("parent_sequences"):
                if isinstance(r["parent_sequences"], list):
                    parent_ids.update(r["parent_sequences"])
                elif r["parent_sequences"]:  # Single ID
                    parent_ids.add(r["parent_sequences"])

        doc_types = set()
        for r in results:
            if r.get("doc_type"):
                doc_types.add(r["doc_type"])

        hint_parts = []
        if doc_types:
            hint_parts.append(
                f"Found {len(results)} documents of types: {', '.join(sorted(doc_types))}"
            )
        else:
            hint_parts.append(f"Found {len(results)} documents")

        if parent_ids:
            id_list = sorted(list(parent_ids))[:3]
            hint_parts.append(
                f"Related to sequences: {id_list}. Search 'pulseq_sequences' for implementations."
            )
        else:
            hint_parts.append(
                "Search 'pulseq_sequences' for related sequence implementations."
            )

        return ". ".join(hint_parts)

    def _format_table_response(
        self,
        table: str,
        results: List[Dict],
        query: str,
        execution_time: int,
        search_type: str = "targeted",
    ) -> Dict:
        """
        Format results according to the standardized table return format using strategy pattern.

        Args:
            table: Table name
            results: Raw results from database
            query: Original query
            execution_time: Time in milliseconds
            search_type: Type of search performed

        Returns:
            Standardized response format
        """
        # Get the formatting strategy for this table
        strategy = self._get_table_formatting_strategy(table)

        # Apply formatting
        formatted_results = strategy["formatter"](results)
        relationships_available = strategy["relationships"](results)
        hint = strategy["hint"](results)

        return {
            "source_table": table,
            "query": query,
            "total_results": len(formatted_results),
            "results": formatted_results,
            "search_metadata": {
                "search_type": search_type,
                "execution_time_ms": execution_time,
                "relationships_available": relationships_available,
                "hint": hint,
            },
        }

    def _format_api_reference_results(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format api_reference table results."""
        formatted = []
        for r in results:
            formatted.append(
                {
                    "function_name": r.get("function_name", ""),
                    "language": r.get("language", "matlab"),
                    "signature": r.get("signature", ""),
                    "description": r.get("description", ""),
                    "calling_pattern": r.get("calling_pattern", ""),
                    "parameters": self._parse_parameters(r.get("parameters", {})),
                    "returns": r.get("returns", {}),
                    "usage_examples": r.get("usage_examples", []),
                    "has_nargin_pattern": r.get("has_nargin_pattern", False),
                    "similarity_score": r.get("similarity", 0.0),
                    "content": r.get(
                        "content", ""
                    ),  # Preserve content field from search
                }
            )
        return formatted

    def _format_pulseq_sequences_results(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format pulseq_sequences table results."""
        formatted = []
        for r in results:
            formatted.append(
                {
                    "id": r.get("id"),
                    "file_name": r.get("file_name", ""),
                    "repository": r.get("repository", ""),
                    "full_code": r.get("full_code", ""),
                    "line_count": r.get("line_count", 0),
                    "language": r.get("language", "matlab"),
                    "content_summary": r.get("content_summary", ""),
                    "sequence_family": r.get("sequence_family", ""),
                    "contrast_mechanism": r.get("contrast_mechanism", ""),
                    "content": r.get(
                        "content", ""
                    ),  # Preserve content field from search
                    "trajectory_type": r.get("trajectory_type", ""),
                    "dimensionality": r.get("dimensionality", ""),
                    "architecture_type": r.get("architecture_type", ""),
                    "complexity_level": r.get("complexity_level", 0),
                    "dependencies": r.get("dependencies", {}),
                    "external_requirements": r.get("external_requirements", []),
                    "preparation_techniques": r.get("preparation_techniques", []),
                    "acceleration_methods": r.get("acceleration_methods", []),
                    "advanced_features": r.get("advanced_features", []),
                    "vendor_features": r.get("vendor_features", []),
                    "vendor_compatibility": r.get("vendor_compatibility", {}),
                    "educational_value": r.get("educational_value", ""),
                    "has_reconstruction_code": r.get("has_reconstruction_code", False),
                    "reconstruction_type": r.get("reconstruction_type", ""),
                    "typical_applications": r.get("typical_applications", []),
                    "typical_scan_time": r.get("typical_scan_time", ""),
                    "tested_vendors": r.get("tested_vendors", []),
                    "all_features": r.get("all_features", []),
                    "similarity_score": r.get("similarity", 0.0),
                }
            )
        return formatted

    def _format_sequence_chunks_results(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format sequence_chunks table results."""
        formatted = []
        for r in results:
            formatted.append(
                {
                    "id": r.get("id"),
                    "sequence_id": r.get("sequence_id"),
                    "chunk_type": r.get("chunk_type", ""),
                    "chunk_order": r.get("chunk_order", 0),
                    "code_content": r.get("code_content", ""),
                    "start_line": r.get("start_line", 0),
                    "end_line": r.get("end_line", 0),
                    "percentage_of_sequence": r.get("percentage_of_sequence", 0),
                    "mri_concept": r.get("mri_concept", ""),
                    "key_concepts": r.get("key_concepts", []),
                    "pulseq_functions": r.get("pulseq_functions", []),
                    "complexity_level": r.get("complexity_level", 0),
                    "description": r.get("description", ""),
                    "parent_context": r.get("parent_context", ""),
                    "similarity_score": r.get("similarity", 0.0),
                    "content": r.get(
                        "content", ""
                    ),  # Preserve content field from search
                }
            )
        return formatted

    def _format_crawled_code_results(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format crawled_code table results."""
        formatted = []
        for r in results:
            formatted.append(
                {
                    "id": r.get("id"),
                    "file_name": r.get("file_name", ""),
                    "source_id": r.get("source_id", ""),
                    "content": r.get("content", ""),
                    "content_type": r.get("content_type", ""),
                    "file_extension": r.get("file_extension", ""),
                    "file_size_bytes": r.get("file_size_bytes", 0),
                    "line_count": r.get("line_count", 0),
                    "parent_sequences": r.get("parent_sequences", []),
                    "dependency_type": r.get("dependency_type", ""),
                    "pulseq_functions_used": r.get("pulseq_functions_used", []),
                    "imports_files": r.get("imports_files", []),
                    "content_summary": r.get("content_summary", ""),
                    "metadata": r.get("metadata", {}),
                    "similarity_score": r.get("similarity", 0.0),
                }
            )
        return formatted

    def _format_crawled_docs_results(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format crawled_docs table results."""
        formatted = []
        for r in results:
            formatted.append(
                {
                    "id": r.get("id"),
                    "resource_uri": r.get("resource_uri", ""),
                    "chunk_number": r.get("chunk_number", 0),
                    "source_id": r.get("source_id", ""),
                    "content": r.get("content", ""),
                    "doc_type": r.get("doc_type", ""),
                    "parent_sequences": r.get("parent_sequences"),
                    "content_summary": r.get("content_summary", ""),
                    "metadata": r.get("metadata", {}),
                    "similarity_score": r.get("similarity", 0.0),
                }
            )
        return formatted

    def _parse_parameters(self, params_data: Any) -> Dict[str, Any]:
        """Parse parameters into required/optional structure matching specification format."""
        if not params_data:
            return {
                "required": [],
                "optional": [],
                "varargin_style": False,
                "varargin_description": "",
            }

        # If it's already structured correctly with required/optional, use it
        if isinstance(params_data, dict) and "required" in params_data:
            # Ensure all fields are present and correctly formatted
            result = {
                "required": [],
                "optional": [],
                "varargin_style": params_data.get("varargin_style", False),
                "varargin_description": params_data.get("varargin_description", ""),
            }

            # Process required parameters
            for param in params_data.get("required", []):
                if isinstance(param, dict):
                    result["required"].append(
                        {
                            "name": str(param.get("name", ""))[:100],
                            "type": str(param.get("type", "unknown"))[:50],
                            "description": str(param.get("description", ""))[:500],
                        }
                    )
                elif isinstance(param, str):
                    result["required"].append(
                        {"name": param, "type": "unknown", "description": ""}
                    )

            # Process optional parameters with defaults
            for param in params_data.get("optional", []):
                if isinstance(param, dict):
                    opt_param = {
                        "name": str(param.get("name", ""))[:100],
                        "type": str(param.get("type", "unknown"))[:50],
                        "description": str(param.get("description", ""))[:500],
                    }
                    # Include default value if present
                    if "default" in param:
                        opt_param["default"] = str(param["default"])[:100]
                    result["optional"].append(opt_param)
                elif isinstance(param, str):
                    result["optional"].append(
                        {"name": param, "type": "unknown", "description": ""}
                    )

            return result

        # Parse from raw parameter data (backward compatibility)
        result = {
            "required": [],
            "optional": [],
            "varargin_style": False,
            "varargin_description": "",
        }

        if not isinstance(params_data, dict):
            logger.warning(f"Expected dict for params_data, got {type(params_data)}")
            return result

        # Check for varargin pattern indicators
        if any(
            key.lower() in ["varargin", "key-value", "kwargs"]
            for key in params_data.keys()
        ):
            result["varargin_style"] = True
            result["varargin_description"] = "Key-value pairs for optional parameters"

        # Parse parameters
        try:
            for key, value in params_data.items():
                if not isinstance(key, str):
                    continue

                # Skip meta keys
                if key in [
                    "varargin_style",
                    "varargin_description",
                    "required",
                    "optional",
                ]:
                    continue

                # Build parameter object matching specification
                param_obj = {
                    "name": str(key)[:100],
                    "type": "unknown",
                    "description": "",
                }

                if isinstance(value, dict):
                    # Extract all available fields
                    param_obj["type"] = str(value.get("type", "unknown"))[:50]
                    param_obj["description"] = str(value.get("description", ""))[:500]

                    # Check if it's required
                    is_required = value.get("required", False) or value.get(
                        "is_required", False
                    )

                    if is_required:
                        result["required"].append(param_obj)
                    else:
                        # Add default value for optional parameters
                        if "default" in value:
                            param_obj["default"] = str(value["default"])[:100]
                        result["optional"].append(param_obj)
                elif isinstance(value, str):
                    # Simple string value - treat as description
                    param_obj["description"] = str(value)[:500]
                    result["optional"].append(param_obj)
                else:
                    # Other types - convert to string
                    param_obj["description"] = str(value)[:500]
                    result["optional"].append(param_obj)

        except Exception as e:
            logger.error(f"Error parsing parameters: {e}")

        return result
