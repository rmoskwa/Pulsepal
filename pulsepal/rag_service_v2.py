"""
Modern RAG Service v2 - Simple retrieval with rich metadata and function validation.

This module provides a simplified RAG service that focuses on fast retrieval
without pattern matching or classification. Intelligence is left to the LLM.
Includes deterministic function validation to prevent hallucinations.
"""

import logging
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from functools import lru_cache
from .supabase_client import get_supabase_client, SupabaseRAGClient
from .settings import get_settings

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

    async def retrieve(
        self, query: str, hint: Optional[RetrievalHint] = None, limit: int = 30
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
            r"(mr|seq|tra|eve|opt)((?:\.aux)?(?:\.quat)?)?\.(\w+)", function_call
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
            is_class_method = correct_data.get("is_class_method", False)
            
            # Extract namespace from correct_usage pattern
            # e.g., "seq.write('filename')" -> "seq"
            # or "mr.makeTrapezoid(...)" -> "mr"
            correct_namespace = None
            if correct_usage:
                ns_match = re.match(r"(mr|seq|tra|eve|opt)(?:\.aux)?(?:\.quat)?", correct_usage)
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
            "explanation": None
        }
        
        # Extract just the method name if it has a namespace
        if '.' in function_name:
            parts = function_name.split('.')
            method = parts[-1]
            namespace = '.'.join(parts[:-1])
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
                    elif "Sequence" in MATLAB_FUNCTIONS.get("class_methods", {}) and \
                         correct in MATLAB_FUNCTIONS["class_methods"]["Sequence"]:
                        result["correct_form"] = f"seq.{correct}"
                    else:
                        result["correct_form"] = correct
                else:
                    result["correct_form"] = correct
                result["explanation"] = f"'{method}' is a common mistake. Use '{correct}' instead."
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
                result["explanation"] = f"'{method}' not found. Did you mean: {', '.join(similar)}?"
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
                "match_threshold": 0.3,
                "match_count": limit // 2,  # Split between sources
            },
        )

        # Search API reference
        api_response = self.supabase_client.rpc(
            "match_api_reference",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.3,
                "match_count": limit // 2,
            },
        )

        # Process crawled pages results
        if hasattr(crawled_response, 'data') and crawled_response.data:
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
        if hasattr(api_response, 'data') and api_response.data:
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
        self, function_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Get exact function documentation from function_calling_patterns view."""
        docs = []

        for func_name in function_names:
            # Handle different namespace patterns
            # Extract just the function name without namespace
            func_match = re.search(
                r"(?:mr|seq|tra|eve|opt)(?:\.aux)?(?:\.quat)?\.(\w+)", func_name
            )
            if func_match:
                clean_func_name = func_match.group(1)
            else:
                clean_func_name = func_name

            response = (
                self.supabase_client.client.table("function_calling_patterns")
                .select(
                    "name, description, correct_usage, usage_instruction, class_name, is_class_method"
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
                ns_match = re.match(r"(mr|seq|tra|eve|opt)(?:\.aux)?(?:\.quat)?", correct_usage)
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

    def _detect_content_type(self, url: str) -> str:
        """Detect content type from URL extension only."""
        if not url:
            return "unknown"
        url_lower = url.lower()
        # Check path patterns first
        if "/examples/" in url_lower or "/demo" in url_lower:
            return "example"
        elif "/api/" in url_lower or "/reference/" in url_lower:
            return "api_reference"
        # Then check extensions
        elif url_lower.endswith(".m"):
            return "matlab_code"
        elif url_lower.endswith(".py"):
            return "python_code"
        elif url_lower.endswith(".md"):
            return "markdown_doc"
        elif url_lower.endswith(".html"):
            return "html_doc"
        else:
            return "documentation"
