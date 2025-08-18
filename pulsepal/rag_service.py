"""
Modern RAG Service v2 - Source-aware retrieval with rich metadata and function validation.

This module provides a source-aware RAG service that intelligently routes queries
to appropriate data sources (API docs, examples, tutorials) based on user intent.
Includes deterministic function validation to prevent hallucinations.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
        self.settings = get_settings()

    @property
    def supabase_client(self) -> SupabaseRAGClient:
        """Lazy load Supabase client."""
        if self._supabase_client is None:
            self._supabase_client = get_supabase_client()
        return self._supabase_client

    async def search_with_source_awareness(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        forced: bool = False,
        source_hints: Optional[Dict] = None,
        detected_functions: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Perform source-aware search across Supabase tables.

        Args:
            query: User's search query
            sources: Specific sources to search (LLM-specified or None for all)
            forced: Whether this search was forced by semantic routing
            source_hints: Additional hints about which sources to prioritize
            detected_functions: Functions detected by semantic router for direct lookup

        Returns:
            Formatted results organized by source with synthesis hints
        """
        # IMPORTANT: Function detection provides hints, not directives
        # Detected functions are passed as metadata to help inform search,
        # but we don't force lookups - let the LLM decide what's needed
        
        # Log detected functions as hints (if any)
        if detected_functions:
            func_names = [f["name"] for f in detected_functions]
            logger.info(f"Function hints available: {len(func_names)} functions detected as context")
            logger.debug(f"Detected functions: {func_names[:5]}...")  # Log first 5 for debugging
            
            # Note: We could optionally do targeted lookups here if the query 
            # specifically asks about one of these functions, but for now
            # we'll let the LLM's search decision take precedence
        
        # If LLM didn't specify sources, default to comprehensive search
        if not sources:
            # Default to all sources with balanced top-k approach
            # This ensures comprehensive coverage without overwhelming the LLM
            sources = ["api_reference", "crawled_pages", "official_sequence_examples"]
            logger.info(f"No sources specified, using top-k from all sources: {sources}")
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
        logger.info(f"Search complete: {total_results} results from {len(source_results)} sources")
        if use_top_k:
            logger.info(f"Used balanced top-k approach: {top_k_per_source} results per source")

        return format_unified_response(source_results, query_context)

    async def retrieve(
        self, query: str, hint: Optional[RetrievalHint] = None, limit: int = 30,
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
            r"(mr|seq|tra|eve|opt)((?:\.aux)?(?:\.quat)?)?\.(\w+)", function_call,
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
                    r"(mr|seq|tra|eve|opt)(?:\.aux)?(?:\.quat)?", correct_usage,
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

    async def _vector_search(self, query: str, limit: int = 30) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
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
                "match_count": limit // 2,  # Split between sources
            },
        ).execute()

        # Search API reference
        api_response = self.supabase_client.rpc(
            "match_api_reference",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.5,  # Balanced threshold to catch semantic matches
                "match_count": limit // 2,
            },
        ).execute()

        # Process crawled pages results
        if hasattr(crawled_response, "data") and crawled_response.data:
            for item in crawled_response.data:
                doc = {
                    "content": item.get("content", ""),
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "similarity": item.get("similarity", 0),
                    "_source": "crawled_pages",
                    "_type": self._detect_content_type(item.get("url", "")),
                }
                results.append(doc)

        # Process API reference results
        if hasattr(api_response, "data") and api_response.data:
            for item in api_response.data:
                doc = {
                    "content": item.get("content", ""),
                    "function_name": item.get("function_name", ""),
                    "category": item.get("category", ""),
                    "similarity": item.get("similarity", 0),
                    "_source": "api_reference",
                    "_type": "api_function",
                }
                results.append(doc)

        return results

    async def _get_function_docs(
        self, function_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Get exact function documentation from function_calling_patterns view."""
        docs = []

        for func_name in function_names:
            # Handle different namespace patterns
            # Extract just the function name without namespace
            func_match = re.search(
                r"(?:mr|seq|tra|eve|opt)(?:\.aux)?(?:\.quat)?\.(\w+)", func_name,
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
                    r"(mr|seq|tra|eve|opt)(?:\.aux)?(?:\.quat)?", correct_usage,
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
                            "function_name": item.get("name", ""),  # Fixed: DB returns 'name' not 'function_name'
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
        self, query: str, limit: int = 5,
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

    async def _direct_function_lookup(self, detected_functions: List[Dict]) -> List[Dict]:
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
                response = self.supabase_client.client.table("api_reference").select(
                    "name, signature, description, "
                    "parameters, returns, usage_examples, "
                    "function_type, class_name, is_class_method, "
                    "calling_pattern, related_functions, "
                    "search_terms, pulseq_version",
                ).ilike("name", func_name).eq("language", "matlab").execute()

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
