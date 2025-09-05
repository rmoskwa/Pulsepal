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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .rag_fusion import merge_search_results, select_top_results
from .reranker_service import BGERerankerService
from .rag_formatters import format_unified_response
from .settings import get_settings
from .source_profiles import MULTI_CHUNK_DOCUMENTS
from .supabase_client import SupabaseRAGClient, get_supabase_client

logger = logging.getLogger(__name__)


@dataclass
class RetrievalHint:
    """Simple hints from LLM about what might be useful."""

    functions_mentioned: List[str] = field(default_factory=list)
    code_provided: bool = False
    search_terms: Optional[List[str]] = None


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
        Now supports parallel BM25 and vector search execution.

        Args:
            query: User's search query
            sources: Specific sources to search (LLM-specified or None for all)
            forced: Whether this search was forced by semantic routing
            source_hints: Additional hints about which sources to prioritize
            detected_functions: Functions detected by semantic router for direct lookup
            use_parallel: Whether to use parallel BM25+vector search (default True)

        Returns:
            Formatted results organized by source with synthesis hints
        """
        # IMPORTANT: Function detection provides hints, not directives
        # Detected functions are passed as metadata to help inform search,
        # but we don't force lookups - let the LLM decide what's needed

        # Log detected functions as hints (if any)
        if detected_functions:
            func_names = [f["name"] for f in detected_functions]
            logger.info(
                f"Function hints available: {len(func_names)} functions detected as context"
            )
            logger.debug(
                f"Detected functions: {func_names[:5]}..."
            )  # Log first 5 for debugging

        # If LLM didn't specify sources, use parallel search for better performance
        if not sources and use_parallel:
            logger.info("Using parallel BM25+vector search for comprehensive results")

            # Execute parallel search
            parallel_results = await self._parallel_search(query, limit=20)

            # Combine results from both search methods
            all_results = []
            all_results.extend(parallel_results.get("bm25_results", []))
            all_results.extend(parallel_results.get("vector_results", []))

            # Deduplicate results based on content similarity
            seen_keys = set()
            unique_results = []
            for result in all_results:
                # Create a key based on source_table and id
                key = f"{result.get('source_table', '')}_{result.get('id', '')}"
                if key not in seen_keys and key != "_":
                    seen_keys.add(key)
                    unique_results.append(result)

            # Format results for unified response
            source_results = {
                "parallel_search": unique_results[:30]
            }  # Limit to 30 total

            # Format results using rag_formatters
            query_context = {
                "original_query": query,
                "forced": forced,
                "source_hints": source_hints,
                "search_mode": "parallel",
                "detected_functions": detected_functions,
                "performance": parallel_results.get("metadata", {}).get(
                    "performance", {}
                ),
            }

            # Log search summary with performance metrics
            perf = parallel_results.get("metadata", {}).get("performance", {})
            logger.info(
                f"Parallel search complete: {len(unique_results)} unique results | "
                f"Latency - BM25: {perf.get('bm25_latency_ms', 0)}ms, "
                f"Vector: {perf.get('vector_latency_ms', 0)}ms, "
                f"Total: {perf.get('total_latency_ms', 0)}ms"
            )

            return format_unified_response(source_results, query_context)

        # Fall back to original source-specific search if sources are specified
        if not sources:
            # Default to all sources with balanced top-k approach
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
            "search_mode": "top_k_balanced" if use_top_k else "targeted",
            "detected_functions": detected_functions,  # Pass as hints, not directives
        }

        # Log search summary
        total_results = sum(len(results) for results in source_results.values())
        logger.info(
            f"Search complete: {total_results} results from {len(source_results)} sources"
        )
        if use_top_k:
            logger.info(
                f"Used balanced top-k approach: {top_k_per_source} results per source"
            )

        return format_unified_response(source_results, query_context)

    async def retrieve(
        self,
        query: str,
        hint: Optional[RetrievalHint] = None,
        limit: int = 30,
    ) -> Dict[str, Any]:
        """
        Simple retrieval based on query and optional hints.
        Returns documents with metadata, no filtering or classification.

        Args:
            query: Search query string
            hint: Optional hints about functions or search terms
            limit: Maximum number of results to return

        Returns:
            Dictionary containing:
            - documents: List of retrieved documents with content and metadata
            - metadata: Overall metadata about the retrieval
        """
        results = {
            "documents": [],
            "metadata": {
                "query": query,
                "total_results": 0,
                "has_function_docs": False,
                "sources": [],
            },
        }

        if hint and hint.functions_mentioned:
            function_docs = await self._get_function_docs(hint.functions_mentioned)
            if function_docs:
                results["documents"].extend(function_docs)
                results["metadata"]["has_function_docs"] = True
                results["metadata"]["sources"].append("function_calling_patterns")

        vector_results = await self._vector_search(query, limit=limit)
        if vector_results:
            results["documents"].extend(vector_results)
            results["metadata"]["sources"].append("vector_search")

        if hint and hint.search_terms:
            for term in hint.search_terms[:3]:
                additional_results = await self._vector_search(term, limit=10)
                if additional_results:
                    results["documents"].extend(additional_results)

        results["documents"].sort(key=lambda x: x.get("similarity", 0), reverse=True)
        results["documents"] = results["documents"][:limit]
        results["metadata"]["total_results"] = len(results["documents"])

        return results

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
        response = (
            self.supabase_client.client.table("function_calling_patterns")
            .select("name, correct_usage, class_name, is_class_method")
            .eq("name", method)
            .execute()
        )

        if response.data:
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
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Execute BM25 and vector searches in parallel using asyncio.gather.

        Args:
            query: Search query text
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
                results = await self._search_bm25_async(query=query, limit=limit)
                bm25_end = time.perf_counter()
                return results
            except Exception as e:
                bm25_end = time.perf_counter()
                logger.error(f"BM25 search failed in parallel execution: {e}")
                return []

        async def timed_vector_search():
            nonlocal vector_start, vector_end
            vector_start = time.perf_counter()
            try:
                results = await self._search_vector_async(query=query, limit=limit)
                vector_end = time.perf_counter()
                return results
            except Exception as e:
                vector_end = time.perf_counter()
                logger.error(f"Vector search failed in parallel execution: {e}")
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
        self, query: str, limit: int = 20, similarity_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Perform async vector similarity search with proper async pattern.

        Args:
            query: Query text for vector search
            limit: Maximum number of results (default 20)
            similarity_threshold: Minimum similarity threshold

        Returns:
            List of vector search results with source attribution
        """
        results = []

        try:
            # Get embedding for query
            from .embeddings import create_embedding

            # Run embedding generation in executor for async
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(None, create_embedding, query)

            # Execute vector searches in parallel
            async def search_crawled_pages():
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

            async def search_api_reference():
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
            logger.error(f"Async vector search failed: {e}")
            return []

    async def _vector_search(self, query: str, limit: int = 30) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        Legacy method kept for backward compatibility.
        """
        # Use the new async method with adjusted limit
        return await self._search_vector_async(query, limit)

    async def _get_function_docs(
        self,
        function_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Get exact function documentation from function_calling_patterns view."""
        docs = []

        for func_name in function_names:
            # Handle different namespace patterns
            # Extract just the function name without namespace
            func_match = re.search(
                r"(?:mr|seq|tra|eve|opt)(?:\.aux)?(?:\.quat)?\.(\w+)",
                func_name,
            )
            if func_match:
                clean_func_name = func_match.group(1)
            else:
                clean_func_name = func_name

            response = (
                self.supabase_client.client.table("function_calling_patterns")
                .select(
                    "name, description, correct_usage, usage_instruction, class_name, is_class_method",
                )
                .eq("name", clean_func_name)
                .execute()
            )

            if response.data:
                for item in response.data:
                    # Check if namespace matches if full function call provided
                    namespace_mismatch = False
                    if func_match:
                        provided_namespace = func_name.split(".")[0]
                        # Extract namespace from correct_usage
                        correct_usage = item.get("correct_usage", "")
                        if correct_usage:
                            ns_match = re.match(r"(mr|seq|tra|eve|opt)", correct_usage)
                            if ns_match and provided_namespace != ns_match.group(1):
                                namespace_mismatch = True

                    doc = {
                        "content": self._format_function_doc(item),
                        "function_name": item.get("name", ""),
                        "correct_usage": item.get("correct_usage", ""),
                        "class_name": item.get("class_name", ""),
                        "similarity": 0.95,  # High score for exact matches
                        "_source": "function_calling_patterns",
                        "_type": "function_doc",
                        "_namespace_mismatch": namespace_mismatch,
                    }
                    docs.append(doc)

        return docs

    def _format_function_doc(self, item: Dict) -> str:
        """Format function documentation for display."""
        parts = []

        if item.get("name"):
            correct_usage = item.get("correct_usage", "")
            if correct_usage:
                # Extract namespace from correct_usage
                ns_match = re.match(
                    r"(mr|seq|tra|eve|opt)(?:\.aux)?(?:\.quat)?",
                    correct_usage,
                )
                if ns_match:
                    parts.append(f"Function: {ns_match.group(0)}.{item['name']}")
                else:
                    parts.append(f"Function: {item['name']}")
            else:
                parts.append(f"Function: {item['name']}")

        if item.get("description"):
            parts.append(f"Description: {item['description']}")

        if item.get("correct_usage"):
            parts.append(f"Usage: {item['correct_usage']}")

        if item.get("usage_instruction"):
            parts.append(f"Instructions: {item['usage_instruction']}")

        return "\n".join(parts)

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
            results = await loop.run_in_executor(
                None,
                self.supabase_client.search_bm25,
                query,
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
        limit: int = 10,
        vector_weight: float = 0.5,
    ) -> List[Dict]:
        """
        Perform hybrid search using parallel BM25 and vector searches with RRF fusion.
        Uses Reciprocal Rank Fusion for improved result merging.

        Args:
            query: Query text
            limit: Maximum number of results
            vector_weight: Weight for vector results (kept for compatibility but not used with RRF)

        Returns:
            List of combined and re-ranked results using RRF
        """
        try:
            # Use the new parallel search implementation
            parallel_results = await self._parallel_search(query, limit=limit * 2)

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
            logger.error(f"Hybrid search failed: {e}")
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
