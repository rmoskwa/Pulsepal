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


def select_top_results(
    fused_results: List[Dict[str, Any]], top_n: int = DEFAULT_TOP_N_RESULTS
) -> List[Dict[str, Any]]:
    """
    Select top N results from fused results, maintaining metadata.

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

    # Ensure all required metadata is present
    for result in top_results:
        # Ensure essential fields exist
        if "title" not in result:
            result["title"] = "Untitled"
        if "content" not in result:
            result["content"] = ""
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
