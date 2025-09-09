"""
Reciprocal Rank Fusion (RRF) algorithm for merging search results.

This module implements RRF to combine results from multiple search methods,
particularly BM25 keyword search and vector semantic search.
"""

import logging
import time
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import hashlib
import statistics

logger = logging.getLogger(__name__)

# Constants for RRF and deduplication
RRF_K_DEFAULT = 60  # Default RRF k parameter (higher = less impact from top ranks)
DEFAULT_TOP_N_RESULTS = 15  # Default number of top results to select
CONTENT_SAMPLE_START = 250  # Characters to sample from content start
CONTENT_SAMPLE_MIDDLE = 250  # Characters to sample from content middle
CONTENT_MIN_LENGTH_FOR_SAMPLING = 500  # Minimum content length to apply sampling


def reciprocal_rank_fusion(
    rankings: List[List[Dict[str, Any]]], k: int = RRF_K_DEFAULT
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Apply Reciprocal Rank Fusion to merge multiple rankings.

    RRF combines multiple rankings using the formula:
    RRF_score(d) = Σ(1/(k + rank_i(d))) for each ranking i

    Args:
        rankings: List of rankings, where each ranking is a list of documents
        k: Constant parameter controlling the impact of high-ranking documents
           (default: 60, higher values decrease impact of top ranks)

    Returns:
        List of tuples containing (document, rrf_score) sorted by score descending
    """
    # Track RRF scores for each unique document
    doc_scores = defaultdict(float)
    doc_data = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            # Create unique identifier for document (using content hash)
            doc_id = _get_document_id(doc)

            # Calculate RRF contribution from this ranking
            rrf_contribution = 1.0 / (k + rank)
            doc_scores[doc_id] += rrf_contribution

            # Store document data (will be overwritten if duplicate, which is fine)
            if doc_id not in doc_data:
                doc_data[doc_id] = doc

    # Combine documents with their final RRF scores
    results = [(doc_data[doc_id], score) for doc_id, score in doc_scores.items()]

    # Sort by RRF score descending
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def merge_search_results(
    bm25_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    k: int = RRF_K_DEFAULT,
) -> List[Dict[str, Any]]:
    """
    Merge BM25 and vector search results using RRF, preserving source attribution.

    Args:
        bm25_results: Results from BM25 keyword search
        vector_results: Results from vector semantic search
        k: RRF parameter (default: 60)

    Returns:
        Merged results with RRF scores and source attribution
    """
    start_time = time.perf_counter()

    # Add source attribution to each result
    for result in bm25_results:
        if "source_methods" not in result:
            result["source_methods"] = []
        result["source_methods"].append("bm25")

    for result in vector_results:
        if "source_methods" not in result:
            result["source_methods"] = []
        result["source_methods"].append("vector")

    # Track duplicates for source attribution
    doc_sources = defaultdict(set)
    all_docs = {}

    # Process BM25 results
    for doc in bm25_results:
        doc_id = _get_document_id(doc)
        doc_sources[doc_id].add("bm25")
        all_docs[doc_id] = doc

    # Process vector results
    for doc in vector_results:
        doc_id = _get_document_id(doc)
        doc_sources[doc_id].add("vector")
        if doc_id in all_docs:
            # Duplicate found - merge source methods
            all_docs[doc_id]["source_methods"] = list(doc_sources[doc_id])
        else:
            all_docs[doc_id] = doc

    # Apply RRF fusion
    fused_results = reciprocal_rank_fusion([bm25_results, vector_results], k=k)

    # Enhance results with complete metadata and source attribution
    final_results = []
    for doc, rrf_score in fused_results:
        doc_id = _get_document_id(doc)
        enhanced_doc = all_docs[doc_id].copy()
        enhanced_doc["rrf_score"] = rrf_score
        enhanced_doc["source_methods"] = list(doc_sources[doc_id])
        final_results.append(enhanced_doc)

    # Log fusion statistics
    fusion_time = time.perf_counter() - start_time
    _log_fusion_statistics(bm25_results, vector_results, final_results, fusion_time)

    return final_results


def _get_complexity_description(level: int) -> str:
    """Convert numeric complexity level to descriptive text."""
    if level is None:
        level = 3  # Default to intermediate if None
    if level <= 2:
        return "beginner-level"
    elif level == 3:
        return "intermediate-level"
    else:
        return "advanced"


def _enhance_content_for_reranker(result: Dict[str, Any], existing_content: str) -> str:
    """
    Enhance existing content with metadata for better reranking.
    This preserves the actual search content while adding contextual information
    that helps the BGE reranker understand complexity and relevance.

    Args:
        result: Search result dictionary with metadata
        existing_content: The actual content from BM25/vector search

    Returns:
        Enhanced content string that preserves original content with metadata prefix
    """
    # Extract metadata with safe defaults
    table_name = result.get("table_name", result.get("source_table", ""))

    # Build metadata prefix based on table type
    metadata_prefix = ""

    if table_name == "pulseq_sequences":
        complexity_level = result.get("complexity_level", 3)
        sequence_family = result.get("sequence_family", "")
        trajectory_type = result.get("trajectory_type", "")

        if sequence_family or trajectory_type:
            complexity_desc = _get_complexity_description(complexity_level)
            metadata_prefix = f"[{complexity_desc} {sequence_family} sequence"
            if trajectory_type:
                metadata_prefix += f", {trajectory_type} trajectory"
            metadata_prefix += "] "

    elif table_name == "api_reference":
        function_name = result.get("function_name", result.get("name", ""))
        function_type = result.get("function_type", "")

        if function_name:
            metadata_prefix = f"[{function_type} function: {function_name}] "

    elif table_name == "sequence_chunks":
        chunk_type = result.get("chunk_type", "")
        mri_concept = result.get("mri_concept", "")

        if chunk_type:
            chunk_type_readable = chunk_type.replace("_", " ")
            metadata_prefix = f"[{chunk_type_readable}"
            if mri_concept:
                metadata_prefix += f", {mri_concept}"
            metadata_prefix += "] "

    elif table_name == "crawled_code":
        content_type = result.get("content_type", "")
        dependency_type = result.get("dependency_type", "")

        if content_type:
            metadata_prefix = f"[{content_type}"
            if dependency_type:
                metadata_prefix += f", {dependency_type} dependency"
            metadata_prefix += "] "

    elif table_name == "crawled_docs":
        doc_type = result.get("doc_type", "")
        if doc_type:
            metadata_prefix = f"[{doc_type} documentation] "

    # Return enhanced content: metadata prefix + original content
    # This gives the reranker context while preserving the actual content
    return metadata_prefix + existing_content


def _create_natural_language_content(result: Dict[str, Any]) -> str:
    """
    Create natural language content for BGE reranker from result metadata.

    BGE reranker works best with natural language, not keywords.
    This function creates descriptive sentences that help the reranker
    understand complexity and pedagogical context.
    """
    # Extract metadata with safe defaults
    table_name = result.get("table_name", result.get("source_table", ""))

    # For pulseq_sequences table, create rich natural language description
    if table_name == "pulseq_sequences":
        complexity_level = result.get("complexity_level", 3)
        sequence_family = result.get("sequence_family", "")
        trajectory_type = result.get("trajectory_type", "")
        content_summary = result.get("content_summary", "")
        educational_value = result.get("educational_value", "")
        typical_applications = result.get("typical_applications", [])
        advanced_features = result.get("advanced_features", [])

        # Build natural language description
        parts = []

        # Complexity and type
        parts.append(
            f"This is a {_get_complexity_description(complexity_level)} {sequence_family} sequence"
        )
        if trajectory_type:
            parts.append(f"that implements {trajectory_type} k-space sampling")

        # Summary if available
        if content_summary:
            parts.append(f". {content_summary}")

        # Applications
        if typical_applications:
            apps_str = ", ".join(typical_applications[:3])  # Limit to 3 for brevity
            parts.append(f". It is designed for {apps_str}")

        # Educational value
        if educational_value:
            parts.append(f"and has {educational_value} educational value")

        # Features
        if advanced_features:
            features_str = ", ".join(advanced_features[:3])  # Limit to 3 for brevity
            parts.append(f". The sequence includes features like {features_str}")

        return " ".join(parts) + "."

    # For api_reference table
    elif table_name == "api_reference":
        function_name = result.get("function_name", result.get("name", ""))
        description = result.get("description", "")
        signature = result.get("signature", "")
        function_type = result.get("function_type", "")
        class_name = result.get("class_name", "")
        is_class_method = result.get("is_class_method", False)
        calling_pattern = result.get("calling_pattern", "")
        parameters = result.get("parameters", {})
        returns = result.get("returns", {})

        # Build natural language description
        parts = []

        # Function context
        if is_class_method and class_name:
            parts.append(f"This is a method of the {class_name} class")
            parts.append(f"named {function_name}")
        else:
            parts.append(f"This is the {function_name} function")

        # Function type
        if function_type and function_type != "main":
            parts.append(f"({function_type} function)")

        # Description
        if description:
            parts.append(f". {description}")

        # Calling pattern if available
        if calling_pattern:
            parts.append(f". Usage: {calling_pattern}")
        elif signature:
            parts.append(f". Signature: {signature}")

        # Parameter information
        if parameters and isinstance(parameters, dict):
            param_count = len(parameters)
            if param_count > 0:
                parts.append(
                    f". Takes {param_count} parameter{'s' if param_count != 1 else ''}"
                )
                # List first few important parameters
                param_names = list(parameters.keys())[:3]
                if param_names:
                    parts.append(f"including {', '.join(param_names)}")

        # Return information
        if returns and isinstance(returns, dict):
            return_type = returns.get("type", "")
            return_desc = returns.get("description", "")
            if return_type:
                parts.append(f". Returns {return_type}")
            elif return_desc:
                parts.append(f". Returns {return_desc}")

        return " ".join(parts) + "."

    # For crawled_code table
    elif table_name == "crawled_code":
        file_name = result.get("file_name", "")
        content_type = result.get("content_type", "")
        dependency_type = result.get("dependency_type", "")
        content_summary = result.get("content_summary", "")
        parent_sequences = result.get("parent_sequences", [])
        pulseq_functions = result.get("pulseq_functions_used", [])
        line_count = result.get("line_count", 0)
        metadata = (
            result.get("metadata", {})
            if isinstance(result.get("metadata"), dict)
            else {}
        )

        # Build natural language description
        parts = []

        # File type and purpose - enhanced with metadata
        if content_type == "helper_function":
            parts.append(f"This is a helper function file {file_name}")
            # Add function category from metadata
            if metadata.get("function_category"):
                parts.append(f"for {metadata['function_category']}")
            if metadata.get("purpose"):
                parts.append(f". Purpose: {metadata['purpose']}")

        elif content_type == "reconstruction":
            parts.append(f"This is a reconstruction code file {file_name}")
            # Add reconstruction type from metadata
            if metadata.get("reconstruction_type"):
                parts.append(
                    f"implementing {metadata['reconstruction_type']} reconstruction"
                )
            if metadata.get("real_time_capable"):
                parts.append("with real-time capability")

        elif content_type == "main_sequence":
            parts.append(f"This is a main sequence file {file_name}")
            # Add sequence type from metadata
            if metadata.get("sequence_type"):
                parts.append(f"implementing a {metadata['sequence_type']} sequence")
            if metadata.get("generates_seq_file"):
                parts.append("that generates .seq files")

        elif content_type == "vendor_conversion_tool":
            parts.append(f"This is a vendor conversion tool {file_name}")
            # Add vendor and conversion details from metadata
            if metadata.get("vendor"):
                parts.append(f"for {metadata['vendor']} scanners")
            if metadata.get("conversion_direction"):
                # Handle conversion direction like "pulseq_to_idea" -> "pulseq to idea"
                direction = metadata["conversion_direction"]
                if "_to_" in direction:
                    # Replace only the middle "_to_" with " to "
                    direction = direction.replace("_to_", " to ")
                else:
                    # Otherwise just replace underscores with spaces
                    direction = direction.replace("_", " ")
                parts.append(f"that converts {direction}")
            if metadata.get("tool_category"):
                parts.append(f"(part of {metadata['tool_category']} framework)")

        elif content_type == "vendor_validation_tool":
            parts.append(f"This is a vendor validation tool {file_name}")
            # Add validation capabilities from metadata
            if metadata.get("vendor"):
                parts.append(f"for {metadata['vendor']} scanners")
            if metadata.get("validation_capabilities"):
                caps = metadata["validation_capabilities"]
                if isinstance(caps, list) and caps:
                    caps_str = ", ".join(caps[:3])  # Limit to 3
                    parts.append(f"that validates {caps_str}")

        elif content_type == "vendor_utility":
            parts.append(f"This is a vendor utility file {file_name}")
            if metadata.get("vendor"):
                parts.append(f"for {metadata['vendor']} systems")
        else:
            parts.append(
                f"This is a {content_type} file {file_name}"
                if content_type
                else f"File {file_name}"
            )

        # Dependency importance
        if dependency_type == "required":
            parts.append("that is required for sequence operation")
        elif dependency_type == "optional":
            parts.append("that provides optional functionality")

        # Summary (if not already added from purpose)
        if content_summary and not metadata.get("purpose"):
            parts.append(f". {content_summary}")

        # Complexity level from metadata
        if metadata.get("complexity_level"):
            level = metadata["complexity_level"]
            if level <= 2:
                parts.append(". This is a beginner-friendly implementation")
            elif level == 3:
                parts.append(". This is an intermediate-level implementation")
            else:
                parts.append(". This is an advanced implementation")

        # Usage context
        if parent_sequences and len(parent_sequences) > 0:
            num_sequences = len(parent_sequences)
            if num_sequences == 1:
                parts.append(". It is used by 1 sequence")
            else:
                parts.append(f". It is used by {num_sequences} different sequences")

        # Key functions from metadata (for vendor tools)
        if metadata.get("key_functions"):
            funcs = metadata["key_functions"]
            if isinstance(funcs, list) and funcs:
                funcs_str = ", ".join(funcs[:3])  # Limit to 3
                parts.append(f". Provides functions: {funcs_str}")
        # Or Pulseq functions used
        elif pulseq_functions and len(pulseq_functions) > 0:
            funcs_str = ", ".join(pulseq_functions[:3])  # Limit to 3
            parts.append(f". Uses Pulseq functions: {funcs_str}")

        # External dependencies from metadata
        if metadata.get("external_dependencies"):
            deps = metadata["external_dependencies"]
            if isinstance(deps, list) and deps:
                deps_str = ", ".join(deps[:2])  # Limit to 2
                parts.append(f". Requires: {deps_str}")

        # Size indicator (if no complexity level provided)
        if not metadata.get("complexity_level") and line_count > 0:
            if line_count < 50:
                parts.append(". This is a small utility")
            elif line_count < 200:
                parts.append(". This is a medium-sized implementation")
            else:
                parts.append(". This is a comprehensive implementation")

        return " ".join(parts) + "."

    # For sequence_chunks table
    elif table_name == "sequence_chunks":
        chunk_type = result.get("chunk_type", "")
        description = result.get("description", "")
        mri_concept = result.get("mri_concept", "")
        pulseq_functions = result.get("pulseq_functions", [])
        key_concepts = result.get("key_concepts", [])
        complexity_level = result.get("complexity_level", 3)
        chunk_order = result.get("chunk_order", 0)
        parent_context = result.get("parent_context", "")

        # Build natural language description
        parts = []

        # Chunk type and context
        if chunk_type:
            chunk_type_readable = chunk_type.replace("_", " ")
            parts.append(f"This is a {chunk_type_readable} code section")
            if chunk_order:
                parts.append(f"(section {chunk_order} in the sequence)")

        # Description
        if description:
            parts.append(f". {description}")

        # MRI concept being demonstrated
        if mri_concept:
            parts.append(f". It demonstrates the MRI physics concept of {mri_concept}")

        # Complexity level
        if complexity_level:
            parts.append(f"at a {_get_complexity_description(complexity_level)} level")

        # Key concepts
        if key_concepts:
            concepts_str = ", ".join(key_concepts[:3])  # Limit to 3 for brevity
            parts.append(f". Key concepts include {concepts_str}")

        # Pulseq functions used
        if pulseq_functions:
            funcs_str = ", ".join(pulseq_functions[:3])  # Limit to 3 for brevity
            parts.append(f". It uses Pulseq functions: {funcs_str}")

        # Parent context if available
        if parent_context:
            parts.append(f". Context: {parent_context}")

        return " ".join(parts) + "."

    # For crawled_docs table
    elif table_name == "crawled_docs":
        doc_type = result.get("doc_type", "")
        source_id = result.get("source_id", "")
        resource_uri = result.get("resource_uri", "")
        content_summary = result.get("content_summary", "")
        parent_sequences = result.get("parent_sequences", [])
        metadata = (
            result.get("metadata", {})
            if isinstance(result.get("metadata"), dict)
            else {}
        )
        chunk_number = result.get("chunk_number", 1)

        # Build natural language description
        parts = []

        # Document type and source
        if doc_type:
            doc_type_readable = doc_type.replace("_", " ")
            parts.append(f"This is a {doc_type_readable} document")
            if chunk_number > 1:
                parts.append(f"(chunk {chunk_number})")
        else:
            parts.append("This is a documentation resource")

        # Add source_id if available, otherwise fall back to resource URI
        if source_id:
            parts.append(f"from {source_id}")
        elif resource_uri:
            # Extract filename or last part of URI for readability
            resource_name = (
                resource_uri.split("/")[-1] if "/" in resource_uri else resource_uri
            )
            parts.append(f"from {resource_name}")

        # Content summary
        if content_summary:
            # Only add period if we have previous content
            if len(parts) > 1:
                parts.append(f". {content_summary}")
            else:
                parts.append(content_summary)

        # Enhanced metadata utilization based on doc_type
        # Different doc_types have different metadata structures per storage-schema.md

        if doc_type in ["vendor_guide", "vendor_troubleshooting", "api_reference"]:
            # Vendor-specific documentation (including api_reference)
            if metadata.get("vendor"):
                parts.append(f". Specific to {metadata['vendor']} scanners")
            if metadata.get("interpreter_version"):
                parts.append(f"for {metadata['interpreter_version']}")
            if metadata.get("topics_covered"):
                # Accept ALL topics, no limiting
                topics = (
                    metadata["topics_covered"]
                    if isinstance(metadata["topics_covered"], list)
                    else []
                )
                if topics:
                    topics_str = ", ".join(str(t).replace("_", " ") for t in topics)
                    parts.append(f". Covers {topics_str}")
            if metadata.get("tool_references"):
                # Accept ALL tool references, no limiting
                tools = (
                    metadata["tool_references"]
                    if isinstance(metadata["tool_references"], list)
                    else []
                )
                if tools:
                    tools_str = ", ".join(tools)
                    parts.append(f". References tools: {tools_str}")
            if doc_type == "vendor_troubleshooting" and metadata.get("error_codes"):
                parts.append(f". Addresses {len(metadata['error_codes'])} error codes")

        elif doc_type == "configuration":
            # Config files - only add config_type
            if metadata.get("config_type"):
                parts.append(f". Config type: {metadata['config_type']}")

        elif doc_type == "paper":
            # Academic papers
            if metadata.get("title"):
                parts.append(f": {metadata['title']}")
            if metadata.get("author"):
                authors = (
                    metadata["author"]
                    if isinstance(metadata["author"], list)
                    else [metadata["author"]]
                )
                authors_str = ", ".join(authors[:2])  # Keep limit for authors
                parts.append(f"by {authors_str}")
            if metadata.get("keywords"):
                keywords = (
                    metadata["keywords"][:3]
                    if isinstance(metadata["keywords"], list)
                    else []
                )  # Keep limit for keywords
                if keywords:
                    parts.append(f". Keywords: {', '.join(keywords)}")

        elif doc_type == "data_file":
            # Data files (CSV, MAT, etc.)
            if metadata.get("data_format"):
                parts.append(f". Format: {metadata['data_format']}")
            if metadata.get("data_type"):
                parts.append(f"containing {metadata['data_type']} data")
            if metadata.get("column_headers"):
                cols = (
                    metadata["column_headers"][:10]
                    if isinstance(metadata["column_headers"], list)
                    else []
                )  # Limit to 10 column headers
                if cols:
                    parts.append(f". Columns include: {', '.join(cols)}")

        elif doc_type in ["tutorial", "guide", "readme", "manual", "forum"]:
            # Educational content and other types - only add topics_covered
            if metadata.get("topics_covered"):
                # Accept ALL topics, no limiting
                topics = (
                    metadata["topics_covered"]
                    if isinstance(metadata["topics_covered"], list)
                    else []
                )
                if topics:
                    topics_str = ", ".join(str(t).replace("_", " ") for t in topics)
                    parts.append(f". Covers {topics_str}")

        # Add parent sequence relationships for any doc_type
        if (
            parent_sequences
            and isinstance(parent_sequences, list)
            and len(parent_sequences) > 0
        ):
            if len(parent_sequences) == 1:
                parts.append(f". Documents sequence ID {parent_sequences[0]}")
            else:
                parts.append(f". Related to {len(parent_sequences)} sequences")

        # Join parts and add final period only if not already present
        result_text = " ".join(parts)
        if result_text and not result_text.endswith("."):
            result_text += "."
        return result_text

    # For other tables or fallback
    else:
        # Use existing content or searchable_text
        content = result.get("content", "")
        if not content:
            content = result.get("searchable_text", "")
        if not content:
            content = result.get("content_summary", "")
        if not content:
            # Last resort: build from available fields
            content = f"{result.get('file_name', '')} {result.get('title', '')}"

        return str(content)


def select_top_results(
    fused_results: List[Dict[str, Any]], top_n: int = DEFAULT_TOP_N_RESULTS
) -> List[Dict[str, Any]]:
    """
    Select top N results from fused results, maintaining metadata.
    Enriches content field with natural language for better reranking.

    Args:
        fused_results: Results after RRF fusion with scores
        top_n: Number of top results to return (default: DEFAULT_TOP_N_RESULTS)

    Returns:
        Top N results sorted by RRF score with normalized scores
    """
    # Already sorted by RRF score from merge_search_results
    top_results = fused_results[:top_n]

    # Normalize scores if results exist
    if top_results:
        max_score = top_results[0].get("rrf_score", 1.0)
        if max_score > 0:
            for result in top_results:
                result["normalized_score"] = result.get("rrf_score", 0) / max_score

    # Ensure all required metadata is present and enrich content for reranker
    for result in top_results:
        # Ensure essential fields exist
        if "title" not in result:
            result["title"] = "Untitled"

        # Create natural language content for BGE reranker in a separate field
        # This is critical for proper reranking with complexity awareness
        # The reranker has a 512 token limit, so we need concise summaries
        # Store in reranker_content field, preserving original content field
        result["reranker_content"] = _create_natural_language_content(result)

        if "source" not in result:
            result["source"] = "unknown"
        if "table_name" not in result:
            result["table_name"] = "unknown"

    return top_results


def _get_document_id(doc: Dict[str, Any]) -> str:
    """
    Generate unique identifier for a document based on its content.

    Improved to handle different document types and avoid over-aggressive deduplication.

    Args:
        doc: Document dictionary

    Returns:
        Unique hash identifier for the document
    """
    # Build a unique key based on available fields
    # Use more content to distinguish between similar chunks
    components = []

    # Add source table if available (different tables might have similar content)
    if "source_table" in doc:
        components.append(f"table:{doc['source_table']}")

    # Add ID if available (most specific identifier)
    if "id" in doc:
        components.append(f"id:{doc['id']}")
    elif "record_id" in doc:
        components.append(f"id:{doc['record_id']}")

    # Add URL if available (for web content)
    if "url" in doc:
        components.append(f"url:{doc['url']}")

    # Add title if available
    if "title" in doc:
        components.append(f"title:{doc['title']}")

    # Add more content for better differentiation
    # This helps distinguish between different chunks of the same document
    content = doc.get("content", "")
    if content:
        # Use more content and include the middle part too (not just the beginning)
        content_sample = (
            content[:CONTENT_SAMPLE_START]
            + content[len(content) // 2 : len(content) // 2 + CONTENT_SAMPLE_MIDDLE]
            if len(content) > CONTENT_MIN_LENGTH_FOR_SAMPLING
            else content
        )
        components.append(f"content:{content_sample}")

    # Add chunk-specific identifiers if present
    if "chunk_id" in doc:
        components.append(f"chunk:{doc['chunk_id']}")
    if "chunk_type" in doc:
        components.append(f"type:{doc['chunk_type']}")

    # Create a unique key from all components
    content_key = "|".join(components) if components else str(doc)
    return hashlib.md5(content_key.encode()).hexdigest()


def _log_fusion_statistics(
    bm25_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    fused_results: List[Dict[str, Any]],
    fusion_time: float,
) -> None:
    """
    Log statistics about the fusion operation.

    Args:
        bm25_results: Original BM25 results
        vector_results: Original vector results
        fused_results: Results after fusion
        fusion_time: Time taken for fusion operation
    """
    # Calculate overlap
    bm25_ids = {_get_document_id(doc) for doc in bm25_results}
    vector_ids = {_get_document_id(doc) for doc in vector_results}
    overlap_count = len(bm25_ids & vector_ids)

    # Calculate score statistics if results exist
    if fused_results:
        scores = [doc.get("rrf_score", 0) for doc in fused_results]
        score_stats = {
            "mean": statistics.mean(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0,
            "min": min(scores),
            "max": max(scores),
        }
    else:
        score_stats = {"mean": 0, "std": 0, "min": 0, "max": 0}

    # Count unique vs duplicate results
    unique_count = len(fused_results)
    total_input = len(bm25_results) + len(vector_results)
    duplicate_count = total_input - unique_count

    logger.info(
        f"RRF Fusion Statistics: "
        f"fusion_time={fusion_time * 1000:.2f}ms, "
        f"bm25_count={len(bm25_results)}, "
        f"vector_count={len(vector_results)}, "
        f"overlap={overlap_count}, "
        f"unique={unique_count}, "
        f"duplicates={duplicate_count}, "
        f"score_mean={score_stats['mean']:.4f}, "
        f"score_std={score_stats['std']:.4f}, "
        f"score_range=[{score_stats['min']:.4f}, {score_stats['max']:.4f}]"
    )
