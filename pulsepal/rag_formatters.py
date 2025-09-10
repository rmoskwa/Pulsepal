"""
Data formatting module for source-aware RAG system.
Formats results from different data sources for optimal LLM consumption.
"""

import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Constants for formatting limits and defaults
CONTENT_PREVIEW_LENGTH = 200  # Characters for content preview
MAX_FUNCTION_HINTS = 10  # Maximum number of function hints to display
DEFAULT_COMPLEXITY_BEGINNER_THRESHOLD = 7000  # Code length threshold for beginner
DEFAULT_COMPLEXITY_INTERMEDIATE_THRESHOLD = (
    15000  # Code length threshold for intermediate
)
DEFAULT_LANGUAGE = "matlab"  # Default programming language
DEFAULT_FUNCTION_TYPE = "main"  # Default function type


def extract_result_metadata(
    result: Dict, source_type: str = "unknown"
) -> Dict[str, Any]:
    """
    Helper function to extract common metadata from various result types.

    Args:
        result: Raw result dictionary
        source_type: Type of source (api_reference, crawled_pages, etc.)

    Returns:
        Dictionary with extracted metadata
    """
    # Extract metadata field if it exists, otherwise use result itself
    metadata = result.get("metadata", {})

    # Handle JSON string metadata (from BM25 results)
    if isinstance(metadata, str):
        try:
            import json

            metadata = json.loads(metadata)
        except (json.JSONDecodeError, ValueError):
            logger.debug("Could not parse metadata JSON string")
            metadata = {}

    # Ensure metadata is a dict
    if not isinstance(metadata, dict):
        metadata = {}

    # Common metadata extraction patterns - check both result and metadata
    return {
        "source_id": result.get("source_id") or metadata.get("source_id", ""),
        "language": result.get("language")
        or metadata.get("language", DEFAULT_LANGUAGE),
        "category": result.get("category") or metadata.get("category", ""),
        "last_updated": result.get("last_updated") or metadata.get("last_updated", ""),
        "repo_name": metadata.get("repo_name", ""),
        "file_path": metadata.get("file_path", ""),
        "source": metadata.get("source", ""),
        "file_extension": metadata.get("file_extension", ""),
        "file_category": metadata.get("file_category", ""),
        "word_count": metadata.get("word_count", 0),
        "all_functions": metadata.get("all_functions", []),
        "dependencies": metadata.get("dependencies", []),
        "truncated": metadata.get("truncated", False),
        # Add more fields commonly found in metadata
        "file_name": result.get("file_name") or metadata.get("file_name", ""),
        "sequence_type": result.get("sequence_type")
        or metadata.get("sequence_type", ""),
        "trajectory_type": result.get("trajectory_type")
        or metadata.get("trajectory_type", ""),
        "repository": result.get("repository") or metadata.get("repository", ""),
        "sequence_family": metadata.get("sequence_family", ""),
        "acceleration": metadata.get("acceleration", ""),
        "ai_summary": metadata.get("ai_summary", ""),
    }


def extract_relevance_score(result: Dict) -> float:
    """
    Extract relevance score from various result formats.

    Args:
        result: Result dictionary

    Returns:
        Relevance score as float
    """
    # Try multiple fields in order of preference
    score_fields = ["_rerank_score", "similarity", "rank", "rrf_score"]

    for field in score_fields:
        if field in result and result[field] is not None:
            return float(result[field])

    return 0.0


def format_api_reference(results: List[Dict]) -> List[Dict]:
    """
    Format API reference data for optimal LLM consumption.
    Emphasizes conceptual understanding followed by technical details.

    Args:
        results: Raw results from API reference search

    Returns:
        Formatted results with structured information
    """
    formatted_results = []

    for result in results:
        # Use helper to extract metadata
        metadata = extract_result_metadata(result, "api_reference")

        formatted_entry = {
            "source_type": "API_DOCUMENTATION",
            "relevance_score": extract_relevance_score(result),
            # Primary Information (what users usually need)
            "function": {
                "name": result.get("function_name", result.get("name", "")),
                "purpose": result.get("description", ""),  # Lead with WHAT it does
                "usage": result.get(
                    "correct_usage",
                    result.get("calling_pattern", ""),
                ),  # HOW to call it correctly
            },
            # Technical Specifications
            "technical_details": {
                "signature": result.get("signature", ""),
                "parameters": format_parameters(result.get("parameters", {})),
                "returns": format_returns(result.get("returns", {})),
                "function_type": result.get("function_type", DEFAULT_FUNCTION_TYPE),
                "class_name": result.get("class_name", ""),
                "is_class_method": result.get("is_class_method", False),
            },
            # Practical Context
            "examples": format_examples(result.get("usage_examples", [])),
            "related_functions": result.get("related_functions", []),
            # Metadata for filtering/context - use extracted metadata
            "metadata": {
                "language": metadata["language"],
                "category": metadata["category"],
                "has_examples": bool(result.get("usage_examples")),
                "last_updated": metadata["last_updated"],
                "source_id": metadata["source_id"],
            },
        }
        formatted_results.append(formatted_entry)

    return formatted_results


def format_parameters(params_json: Union[Dict[str, Any], List[Any], str, None]) -> str:
    """
    Format parameters for clear presentation with input validation.

    Args:
        params_json: Parameters in JSON format

    Returns:
        Formatted parameter string
    """
    if not params_json:
        return "No parameters"

    # Handle string parameters (might be JSON string)
    if isinstance(params_json, str):
        try:
            import json

            params_json = json.loads(params_json)
            logger.debug(
                "Successfully parsed parameters JSON string",
                extra={"input_length": len(params_json)},
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "Failed to parse parameters JSON, returning as-is",
                extra={
                    "error": str(e),
                    "input_sample": params_json[:100] if params_json else "",
                },
            )
            return params_json

    # Validate expected types
    if not isinstance(params_json, (dict, list)):
        logger.warning(
            "Unexpected parameter type, converting to string",
            extra={
                "type": type(params_json).__name__,
                "value_sample": str(params_json)[:100] if params_json else "",
            },
        )
        return str(params_json)

    # If it's a dictionary, format each parameter
    if isinstance(params_json, dict):
        formatted_params = []
        for param_name, param_info in params_json.items():
            # Validate param_name is string
            if not isinstance(param_name, str):
                logger.warning(
                    "Non-string parameter name encountered",
                    extra={
                        "name_type": type(param_name).__name__,
                        "value": str(param_name),
                    },
                )
                param_name = str(param_name)

            param_str = f"• {param_name}"

            if isinstance(param_info, dict):
                if param_info.get("type"):
                    param_str += f" ({param_info['type']})"
                if param_info.get("required"):
                    param_str += " [REQUIRED]"
                else:
                    param_str += " [optional]"
                if param_info.get("default"):
                    param_str += f" = {param_info['default']}"
                if param_info.get("description"):
                    param_str += f"\n    {param_info['description']}"
            else:
                param_str += f": {param_info}"

            formatted_params.append(param_str)

        return "\n".join(formatted_params)

    # If it's a list, format as list
    if isinstance(params_json, list):
        return "\n".join(f"• {param}" for param in params_json)


def format_returns(returns_json: Union[Dict[str, Any], str, None]) -> str:
    """
    Format return values for clarity with input validation.

    Args:
        returns_json: Return value specification in JSON format

    Returns:
        Formatted return value string
    """
    if not returns_json:
        return "No return value"

    # Handle string returns
    if isinstance(returns_json, str):
        try:
            import json

            returns_json = json.loads(returns_json)
            logger.debug(
                "Successfully parsed returns JSON string",
                extra={"input_length": len(str(returns_json))},
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "Failed to parse returns JSON, returning as-is",
                extra={
                    "error": str(e),
                    "input_sample": returns_json[:100] if returns_json else "",
                },
            )
            return returns_json

    # Validate expected type
    if not isinstance(returns_json, dict):
        logger.warning(
            "Unexpected returns type, converting to string",
            extra={
                "type": type(returns_json).__name__,
                "value_sample": str(returns_json)[:100] if returns_json else "",
            },
        )
        return str(returns_json)

    # Format dict returns with validation
    return_str = returns_json.get("type", "unknown")
    if returns_json.get("description"):
        description = returns_json.get("description")
        if not isinstance(description, str):
            logger.warning(
                "Non-string return description",
                extra={"desc_type": type(description).__name__},
            )
            description = str(description)
        return_str += f" - {description}"

    return return_str


def format_examples(
    examples_json: Union[List[Dict[str, Any]], Dict[str, Any], str, None],
) -> List[Dict[str, str]]:
    """
    Format usage examples with language tags and input validation.

    Args:
        examples_json: Examples in JSON format

    Returns:
        List of formatted example dictionaries
    """
    if not examples_json:
        return []

    # Handle string examples
    if isinstance(examples_json, str):
        try:
            import json

            examples_json = json.loads(examples_json)
            logger.debug(
                "Successfully parsed examples JSON string",
                extra={"result_type": type(examples_json).__name__},
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "Failed to parse examples JSON, treating as code string",
                extra={
                    "error": str(e),
                    "input_sample": examples_json[:100] if examples_json else "",
                },
            )
            return [{"code": examples_json, "language": DEFAULT_LANGUAGE}]

    # Ensure we have a list
    if not isinstance(examples_json, list):
        if isinstance(examples_json, dict):
            logger.debug("Converting single example dict to list")
            examples_json = [examples_json]
        else:
            logger.warning(
                "Unexpected examples type, converting to single example",
                extra={
                    "type": type(examples_json).__name__,
                    "value_sample": str(examples_json)[:100] if examples_json else "",
                },
            )
            examples_json = [{"code": str(examples_json), "language": DEFAULT_LANGUAGE}]

    formatted_examples = []
    for idx, example in enumerate(examples_json):
        if isinstance(example, dict):
            # Validate and extract fields with type checking
            code = example.get("code", "")
            if not isinstance(code, str):
                logger.warning(
                    f"Non-string code in example {idx}",
                    extra={"code_type": type(code).__name__},
                )
                code = str(code)

            description = example.get("description", "")
            if description and not isinstance(description, str):
                logger.warning(
                    f"Non-string description in example {idx}",
                    extra={"desc_type": type(description).__name__},
                )
                description = str(description)

            language = example.get("language", DEFAULT_LANGUAGE)
            if not isinstance(language, str):
                logger.warning(
                    f"Non-string language in example {idx}",
                    extra={"lang_type": type(language).__name__},
                )
                language = DEFAULT_LANGUAGE

            formatted_examples.append(
                {
                    "code": code,
                    "description": description,
                    "language": language,
                }
            )
        else:
            logger.debug(
                f"Converting non-dict example {idx} to string",
                extra={"type": type(example).__name__},
            )
            formatted_examples.append(
                {"code": str(example), "language": DEFAULT_LANGUAGE}
            )

    return formatted_examples


def format_crawled_pages(results: List[Dict]) -> List[Dict]:
    """
    Format crawled pages with special handling for multi-chunk documents.
    Groups chunks from the same document together.

    Args:
        results: Raw results from crawled pages search

    Returns:
        Formatted results with document grouping
    """
    # First, group results by URL to handle multi-chunk documents
    documents_by_url = {}
    for result in results:
        url = result.get("url", "")
        if not url:
            # Handle results without URL
            url = f"unknown_{id(result)}"

        if url not in documents_by_url:
            documents_by_url[url] = []
        documents_by_url[url].append(result)

    formatted_results = []

    for url, chunks in documents_by_url.items():
        # Sort chunks by chunk_number to maintain document order
        chunks.sort(key=lambda x: x.get("chunk_number", 0))

        # Determine if this is a multi-chunk document
        is_multi_chunk = len(chunks) > 1

        if is_multi_chunk:
            # Combine chunks for coherent presentation
            formatted_entry = format_multi_chunk_document(url, chunks)
        else:
            # Format single chunk normally
            formatted_entry = format_single_chunk(chunks[0])

        formatted_results.append(formatted_entry)

    return formatted_results


def format_multi_chunk_document(url: str, chunks: List[Dict]) -> Dict:
    """
    Format multi-chunk documents with context preservation.

    Args:
        url: Document URL
        chunks: List of document chunks

    Returns:
        Formatted multi-chunk document
    """
    total_chunks = len(chunks)

    # Combine content with chunk markers
    combined_content = []
    for chunk in chunks:
        chunk_num = chunk.get("chunk_number", 0)
        content = chunk.get("content", "")

        if total_chunks > 1:
            combined_content.append(f"[Part {chunk_num + 1}/{total_chunks}]")
        combined_content.append(content)

    # Extract common metadata from first chunk using helper
    first_chunk = chunks[0]
    metadata = extract_result_metadata(first_chunk, "crawled_pages")

    return {
        "source_type": "DOCUMENTATION_MULTI_PART",
        "relevance_score": max(extract_relevance_score(chunk) for chunk in chunks),
        "document": {
            "url": url,
            "title": extract_title_from_url(url) if url != "unknown" else "Document",
            "total_parts": total_chunks,
            "content": "\n\n".join(combined_content),
            "content_length": sum(len(chunk.get("content", "")) for chunk in chunks),
        },
        "context": {
            "source": metadata["source"],
            "file_path": metadata["file_path"],
            "language": metadata["language"],
            "document_type": get_document_type(url, metadata),
        },
        "metadata": {
            "source_id": metadata["source_id"],
            "repo_name": metadata["repo_name"],
            "functions_mentioned": metadata["all_functions"],
            "dependencies": metadata["dependencies"],
            "is_official": "pulseq.github.io" in url if url else False,
        },
        "retrieval_note": f"Retrieved all {total_chunks} parts for document coherence",
    }


def format_single_chunk(chunk: Dict) -> Dict:
    """
    Format single chunk documents.

    Args:
        chunk: Single document chunk

    Returns:
        Formatted single chunk document
    """
    # Use helper to extract metadata
    metadata = extract_result_metadata(chunk, "crawled_pages")
    url = chunk.get("url", "")

    return {
        "source_type": "CODE_EXAMPLE" if is_code(chunk) else "DOCUMENTATION",
        "relevance_score": extract_relevance_score(chunk),
        "content": {
            "text": chunk.get("content", ""),
            "url": url,
            "title": extract_title_from_url(url)
            if url
            else chunk.get("title", "Untitled"),
        },
        "context": {
            "source": metadata["source"],
            "file_path": metadata["file_path"],
            "language": metadata["language"],
            "file_type": metadata["file_extension"],
        },
        "code_info": {
            "functions_used": metadata["all_functions"],
            "dependencies": metadata["dependencies"],
            "is_complete": not metadata["truncated"],
        },
        "metadata": {
            "source_id": metadata["source_id"],
            "repo_name": metadata["repo_name"],
            "word_count": metadata["word_count"],
            "is_official": "pulseq.github.io" in url if url else False,
        },
    }


def is_code(chunk: Dict) -> bool:
    """
    Determine if chunk is primarily code.

    Args:
        chunk: Document chunk

    Returns:
        True if chunk is primarily code
    """
    # Use helper to extract metadata
    metadata = extract_result_metadata(chunk, "crawled_pages")
    content = chunk.get("content", "")

    # Check file extension
    if metadata["file_extension"] in [".m", ".py", ".cpp", ".c", ".h"]:
        return True

    # Check file category
    if metadata["file_category"] == "repository_file":
        return True

    # Check content patterns (simple heuristic)
    code_patterns = ["function ", "def ", "class ", "#include", "import ", "//"]
    code_count = sum(1 for pattern in code_patterns if pattern in content)

    return code_count >= 2


def extract_title_from_url(url: str) -> str:
    """
    Extract a readable title from URL with input validation.

    Args:
        url: URL string

    Returns:
        Readable title
    """
    if not url:
        return "Untitled"

    # Validate input type
    if not isinstance(url, str):
        logger.warning(
            "Non-string URL passed to extract_title_from_url",
            extra={"type": type(url).__name__, "value": str(url)[:100]},
        )
        url = str(url)

    if url.startswith("unknown"):
        return "Untitled"

    # Special handling for known patterns
    if "specification.pdf" in url:
        return "Pulseq Specification Document"

    # Extract filename or last path segment
    parsed = urlparse(url)
    path = parsed.path

    if path:
        filename = os.path.basename(path)
        # Remove extension and convert to title case
        title = os.path.splitext(filename)[0]
        # Clean up common prefixes
        for prefix in ["write", "demo", "test"]:
            if title.lower().startswith(prefix):
                title = title[len(prefix) :]
        # Convert to readable format
        title = title.replace("_", " ").replace("-", " ")
        # Handle camelCase
        title = re.sub(r"([a-z])([A-Z])", r"\1 \2", title)
        return title.strip().title() if title.strip() else "Document"

    return parsed.netloc or "Document"


def get_document_type(url: str, metadata: Dict[str, Any]) -> str:
    """
    Determine document type from URL and metadata.

    Args:
        url: Document URL
        metadata: Document metadata

    Returns:
        Document type string
    """
    if not url:
        return "unknown"

    url_lower = url.lower()

    # Check by extension
    if url_lower.endswith(".pdf"):
        return "pdf_specification"
    if url_lower.endswith(".html"):
        return "html_documentation"
    if url_lower.endswith(".md"):
        return "markdown_documentation"
    if url_lower.endswith(".m"):
        return "matlab_code"
    if url_lower.endswith(".py"):
        return "python_code"

    # Check by path patterns
    if "/examples/" in url_lower or "/demo" in url_lower:
        return "example_code"
    if "/api/" in url_lower or "/reference/" in url_lower:
        return "api_documentation"
    if "github.com" in url_lower:
        return "github_repository"

    return metadata.get("file_category", "documentation")


def format_official_sequence_examples(results: List[Dict]) -> List[Dict]:
    """
    Format official sequence examples as educational tutorials.
    Emphasizes learning value and practical implementation.

    Args:
        results: Raw results from official sequence examples search

    Returns:
        Formatted educational sequence results
    """
    formatted_results = []

    for result in results:
        # Use helper to extract common metadata
        metadata = extract_result_metadata(result, "official_sequences")

        # Extract sequence-specific fields
        file_name = result.get("file_name") or metadata.get("file_name", "")
        sequence_type = result.get("sequence_type") or metadata.get("sequence_type", "")
        trajectory_type = result.get("trajectory_type") or metadata.get(
            "trajectory_type", ""
        )
        acceleration = result.get("acceleration") or metadata.get("acceleration", "")
        repository = result.get("repository") or metadata.get("repository", "")
        sequence_family = metadata.get("sequence_family", "")

        # Parse the AI summary for key insights
        summary_sections = parse_ai_summary(
            result.get("ai_summary", metadata.get("ai_summary", ""))
        )

        formatted_entry = {
            "source_type": "EDUCATIONAL_SEQUENCE",
            "relevance_score": extract_relevance_score(result),
            # Educational Context (PRIMARY)
            "tutorial_info": {
                "title": extract_sequence_title(file_name),
                "sequence_type": sequence_type or sequence_family or "Unknown",
                "complexity": determine_complexity(result),
                "summary": summary_sections.get("overview", "")
                or result.get("content", "")[:CONTENT_PREVIEW_LENGTH],
                "key_techniques": summary_sections.get("techniques", []),
                "learning_points": summary_sections.get("learning_points", []),
            },
            # Complete Implementation
            "implementation": {
                "full_code": result.get("content", "") or result.get("full_code", ""),
                "code_length": len(
                    result.get("content", "") or result.get("full_code", "")
                ),
                "language": "MATLAB",  # All official examples are MATLAB
            },
            # Technical Specifications
            "technical_details": {
                "trajectory_type": trajectory_type,
                "acceleration": acceleration or "none",
                "performance_notes": summary_sections.get("performance", ""),
                "warnings": summary_sections.get("warnings", []),
            },
            # Usage Context
            "usage_guide": {
                "when_to_use": generate_usage_context(result),
                "prerequisites": extract_prerequisites(result),
                "modifications_suggested": summary_sections.get("customization", []),
            },
            # Metadata
            "metadata": {
                "file_name": file_name,
                "repository": repository,
                "is_demo": "demo" in file_name.lower(),
                "validation_status": "official_tested",
                "suitable_for_learning": True,
            },
        }
        formatted_results.append(formatted_entry)

    return formatted_results


def parse_ai_summary(summary: str) -> Dict[str, Union[str, List[str]]]:
    """
    Parse AI summary into structured sections with input validation.

    Args:
        summary: AI-generated summary text

    Returns:
        Dictionary of parsed sections
    """
    sections = {
        "overview": "",
        "techniques": [],
        "learning_points": [],
        "performance": "",
        "warnings": [],
        "customization": [],
    }

    if not summary:
        return sections

    # Validate input type
    if not isinstance(summary, str):
        logger.warning(
            "Non-string AI summary, converting to string",
            extra={
                "type": type(summary).__name__,
                "value_sample": str(summary)[:100] if summary else "",
            },
        )
        summary = str(summary)

    # Extract overview (usually first paragraph)
    paragraphs = summary.split("\n\n")
    if paragraphs:
        sections["overview"] = paragraphs[0]

    # Look for key phrases
    summary_lower = summary.lower()

    # Extract techniques mentioned
    if "technique" in summary_lower or "method" in summary_lower:
        # Parse for technical terms
        techniques = extract_technical_terms(summary)
        sections["techniques"] = techniques

    # Extract warnings
    if (
        "warning" in summary_lower
        or "caution" in summary_lower
        or "note" in summary_lower
    ):
        sections["warnings"].append("Contains important notes - review carefully")

    # Extract performance notes
    if "performance" in summary_lower or "optimiz" in summary_lower:
        sections["performance"] = "Includes performance optimizations"

    # Extract learning points from bullet points or numbered lists
    lines = summary.split("\n")
    for line in lines:
        if line.strip().startswith(("•", "-", "*", "1.", "2.", "3.")):
            sections["learning_points"].append(line.strip().lstrip("•-*123456789. "))

    return sections


def extract_technical_terms(text: str) -> List[str]:
    """
    Extract technical MRI/Pulseq terms from text.

    Args:
        text: Text to extract terms from

    Returns:
        List of technical terms
    """
    technical_patterns = [
        r"\b(?:EPI|TSE|GRE|MPRAGE|FLASH|HASTE|TruFISP|SSFP)\b",
        r"\b(?:gradient|RF|pulse|echo|spin|excitation|refocusing)\b",
        r"\b(?:k-space|trajectory|readout|phase encoding)\b",
        r"\b(?:TE|TR|flip angle|bandwidth|FOV)\b",
    ]

    terms = set()
    for pattern in technical_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        terms.update(matches)

    return list(terms)


def determine_complexity(result: Dict) -> str:
    """
    Determine complexity level based on various factors with input validation.

    Args:
        result: Sequence result dictionary

    Returns:
        Complexity level string
    """
    # Validate input
    if not isinstance(result, dict):
        logger.warning(
            "Non-dict result passed to determine_complexity",
            extra={"type": type(result).__name__},
        )
        return "unknown"

    content = result.get("content", "")
    if not isinstance(content, str):
        logger.warning(
            "Non-string content in complexity determination",
            extra={"content_type": type(content).__name__},
        )
        content = str(content) if content else ""

    content_length = len(content)
    sequence_type = result.get("sequence_type", "")

    if not isinstance(sequence_type, str):
        logger.warning(
            "Non-string sequence_type in complexity determination",
            extra={"type": type(sequence_type).__name__},
        )
        sequence_type = str(sequence_type) if sequence_type else ""

    # Simple heuristic based on length and type
    if content_length < DEFAULT_COMPLEXITY_BEGINNER_THRESHOLD:
        complexity = "beginner"
    elif content_length < DEFAULT_COMPLEXITY_INTERMEDIATE_THRESHOLD:
        complexity = "intermediate"
    else:
        complexity = "advanced"

    # Adjust for specific types known to be complex
    if sequence_type in ["Diffusion", "EPI", "Spiral"]:
        if complexity == "beginner":
            complexity = "intermediate"
        elif complexity == "intermediate":
            complexity = "advanced"

    return complexity


def extract_sequence_title(file_name: str) -> str:
    """
    Extract readable title from filename.

    Args:
        file_name: Sequence file name

    Returns:
        Readable title
    """
    if not file_name:
        return "Sequence Example"

    # Remove path and extension
    base_name = os.path.splitext(os.path.basename(file_name))[0]

    # Remove common prefixes
    for prefix in ["write", "demo", "test", "example"]:
        if base_name.lower().startswith(prefix):
            base_name = base_name[len(prefix) :]

    # Convert to readable format
    # writeEpiDiffusionRS_PMC -> EPI Diffusion RS PMC
    title = re.sub(r"([a-z])([A-Z])", r"\1 \2", base_name)
    title = title.replace("_", " - ")

    return title.strip() or "Sequence Example"


def generate_usage_context(result: Dict) -> str:
    """
    Generate usage context based on sequence characteristics.

    Args:
        result: Sequence result dictionary

    Returns:
        Usage context string
    """
    sequence_type = result.get("sequence_type", "").lower()
    trajectory = result.get("trajectory_type", "").lower()

    contexts = {
        "epi": "Fast imaging applications, fMRI, diffusion imaging",
        "tse": "T2-weighted imaging with reduced scan time",
        "mprage": "High-resolution T1-weighted brain imaging",
        "diffusion": "Diffusion-weighted imaging, DTI, DWI",
        "spiral": "Fast imaging with efficient k-space coverage",
        "radial": "Motion-robust imaging, reduced artifacts",
        "gre": "T2*-weighted imaging, susceptibility contrast",
    }

    for key, context in contexts.items():
        if key in sequence_type or key in trajectory:
            return context

    return "General MRI sequence development and testing"


def extract_prerequisites(result: Dict) -> List[str]:
    """
    Extract prerequisites based on sequence complexity.

    Args:
        result: Sequence result dictionary

    Returns:
        List of prerequisites
    """
    complexity = determine_complexity(result)
    sequence_type = result.get("sequence_type", "")

    prerequisites = ["Basic understanding of MRI physics"]

    if complexity in ["intermediate", "advanced"]:
        prerequisites.append("Familiarity with Pulseq framework")
        prerequisites.append("Understanding of k-space trajectories")

    if complexity == "advanced":
        prerequisites.append("Experience with pulse sequence programming")

    if "diffusion" in sequence_type.lower():
        prerequisites.append("Knowledge of diffusion MRI principles")

    if "epi" in sequence_type.lower():
        prerequisites.append("Understanding of EPI readout strategies")

    return prerequisites


def format_unified_response(
    source_results: Dict[str, List],
    query_context: Dict,
) -> Dict:
    """
    Create a unified response structure that Gemini can easily process.
    Handles mixed sources and provides clear context.

    Args:
        source_results: Dictionary of results by source type
        query_context: Context about the original query

    Returns:
        Unified response dictionary
    """
    response = {
        "query": query_context.get("original_query", ""),
        "search_metadata": {
            "sources_searched": list(source_results.keys()),
            "total_results": sum(len(results) for results in source_results.values()),
            "timestamp": datetime.now().isoformat(),
        },
        "results_by_source": {},
        "synthesis_hints": [],
        "citation_map": {},
    }

    # Pass through performance metrics including rerank stats
    if "performance" in query_context:
        response["search_metadata"]["performance"] = query_context["performance"]
    if "rerank_stats" in query_context:
        response["search_metadata"]["rerank_stats"] = query_context["rerank_stats"]

    # Function detection is no longer passed to formatters

    # Process each source type
    for source_type, results in source_results.items():
        if source_type == "api_reference":
            formatted = format_api_reference(results)
            response["results_by_source"]["api_reference"] = formatted
            response["synthesis_hints"].append(
                "API documentation provides authoritative function specifications",
            )

        elif source_type == "crawled_pages":
            formatted = format_crawled_pages(results)
            response["results_by_source"]["examples_and_docs"] = formatted
            response["synthesis_hints"].append(
                "Examples show practical implementation patterns",
            )

        elif source_type == "official_sequence_examples":
            formatted = format_official_sequence_examples(results)
            response["results_by_source"]["tutorials"] = formatted
            response["synthesis_hints"].append(
                "Official sequences are tested and educational",
            )

        elif source_type in ["hybrid_search", "parallel_search"]:
            # Handle hybrid search results - group by source table
            grouped_results = {}
            for result in results:
                source_table = result.get("source_table", "unknown")
                if source_table not in grouped_results:
                    grouped_results[source_table] = []
                grouped_results[source_table].append(result)

            # Format each group appropriately
            for table, table_results in grouped_results.items():
                if table == "api_reference":
                    formatted = format_api_reference(table_results)
                    response["results_by_source"]["api_reference"] = formatted
                elif table in ["crawled_docs", "crawled_pages", "crawled_code"]:
                    formatted = format_crawled_pages(table_results)
                    response["results_by_source"]["examples_and_docs"] = formatted
                elif table in ["pulseq_sequences", "sequence_chunks"]:
                    formatted = format_official_sequence_examples(table_results)
                    response["results_by_source"]["tutorials"] = formatted
                else:
                    # Generic formatting for unknown sources
                    formatted = table_results
                    response["results_by_source"][table] = formatted

            response["synthesis_hints"].append(
                "Results from hybrid search combining keyword and semantic relevance",
            )

    # Add citation guidance
    response["citation_map"] = generate_citation_map(response["results_by_source"])

    # Add synthesis recommendations
    response["synthesis_recommendations"] = generate_synthesis_recommendations(
        source_results,
        query_context,
    )

    return response


def generate_citation_map(results_by_source: Dict) -> Dict[str, str]:
    """
    Create a map for proper citation of sources.

    Args:
        results_by_source: Results organized by source type

    Returns:
        Citation map dictionary
    """
    citation_map = {}
    citation_id = 1

    for source_type, results in results_by_source.items():
        for result in results:
            citation_key = f"[{citation_id}]"

            if source_type == "api_documentation":
                function_name = result.get("function", {}).get("name", "Unknown")
                citation_text = f"{function_name} - Pulseq API Documentation"

            elif source_type == "examples_and_docs":
                if "document" in result:
                    doc_title = result["document"].get("title", "Unknown")
                else:
                    doc_title = result.get("content", {}).get("title", "Unknown")
                citation_text = f"{doc_title}"

            elif source_type == "tutorials":
                tutorial_title = result.get("tutorial_info", {}).get("title", "Unknown")
                citation_text = f"{tutorial_title} - Official Example"
            else:
                citation_text = "Source Document"

            citation_map[citation_key] = citation_text
            citation_id += 1

    return citation_map


def generate_synthesis_recommendations(
    source_results: Dict,
    query_context: Dict,
) -> List[str]:
    """
    Generate recommendations for how to synthesize the results.

    Args:
        source_results: Results by source type
        query_context: Query context information

    Returns:
        List of synthesis recommendations
    """
    recommendations = []

    # Check what types of results we have
    has_api = "api_reference" in source_results and source_results["api_reference"]
    has_examples = "crawled_pages" in source_results and source_results["crawled_pages"]
    has_tutorials = (
        "official_sequence_examples" in source_results
        and source_results["official_sequence_examples"]
    )

    # Query type detection
    query_lower = query_context.get("original_query", "").lower()
    is_learning = any(
        word in query_lower
        for word in ["learn", "tutorial", "how to", "teach", "example"]
    )
    is_parameter = any(
        word in query_lower
        for word in ["parameter", "argument", "signature", "input", "output"]
    )
    is_debug = any(
        word in query_lower for word in ["error", "wrong", "fix", "debug", "issue"]
    )

    # Generate specific recommendations
    if is_learning and has_tutorials:
        recommendations.append("Start with official sequence examples for learning")
        recommendations.append("Reference API documentation for parameter details")

    elif is_parameter and has_api:
        recommendations.append("Focus on API documentation for accurate parameters")
        recommendations.append("Include examples if available for context")

    elif is_debug:
        recommendations.append("Check API documentation for correct usage")
        recommendations.append("Compare with working examples")

    else:
        recommendations.append("Synthesize information from all available sources")

    # Add source-specific recommendations
    if has_api and has_examples:
        recommendations.append(
            "Combine API specs with practical examples for complete understanding",
        )

    if has_tutorials:
        recommendations.append(
            "Official examples provide best practices and tested implementations",
        )

    return recommendations


def format_parameters_comprehensive(params_json: Any) -> str:
    """
    Format parameters with complete details for direct lookup results.

    Args:
        params_json: Parameters in JSON format

    Returns:
        Formatted parameter string with comprehensive details
    """

    if not params_json:
        return "No parameters"

    # Handle string parameters
    if isinstance(params_json, str):
        try:
            import json

            params_json = json.loads(params_json)
        except (json.JSONDecodeError, ValueError):
            return params_json

    # Format based on structure
    if isinstance(params_json, dict):
        lines = []

        # Handle required/optional structure
        required = params_json.get("required", [])
        optional = params_json.get("optional", [])

        if required:
            lines.append("**Required Parameters:**")
            for param in required:
                lines.append(format_single_parameter(param, is_required=True))

        if optional:
            if lines:
                lines.append("")  # Add spacing
            lines.append("**Optional Parameters:**")
            for param in optional:
                lines.append(format_single_parameter(param, is_required=False))

        # Handle flat parameter structure
        if not required and not optional:
            for param_name, param_info in params_json.items():
                if param_name not in ["required", "optional"]:
                    lines.append(
                        format_single_parameter(
                            {"name": param_name, **param_info}
                            if isinstance(param_info, dict)
                            else {"name": param_name, "description": str(param_info)},
                        )
                    )

        return "\n".join(lines) if lines else "No parameters specified"

    return str(params_json)


def format_single_parameter(param: Dict, is_required: bool = None) -> str:
    """
    Format a single parameter with all available details.

    Args:
        param: Parameter dictionary
        is_required: Whether parameter is required

    Returns:
        Formatted parameter string
    """
    if isinstance(param, str):
        return f"• {param}"

    parts = []
    name = param.get("name", "unknown")
    param_type = param.get("type", "")
    units = param.get("units", "")
    default = param.get("default", "")
    description = param.get("description", "")
    example = param.get("example", "")
    valid_values = param.get("valid_values", "")

    # Build parameter header
    header = f"• **`{name}`**"

    if param_type:
        header += f" ({param_type})"

    if units and units != "none":
        header += f" [{units}]"

    if is_required is not None:
        header += " - REQUIRED" if is_required else " - optional"

    parts.append(header)

    # Add details with proper indentation
    if description:
        parts.append(f"  - Description: {description}")

    if default and default not in ["[]", "0", "none"]:
        parts.append(f"  - Default: `{default}`")

    if example:
        parts.append(f"  - Example: `{example}`")

    if valid_values:
        parts.append(f"  - Valid values: {valid_values}")

    return "\n".join(parts)


def format_returns_comprehensive(returns_json: Any) -> str:
    """
    Format return values comprehensively.

    Args:
        returns_json: Returns in JSON format

    Returns:
        Formatted returns string
    """
    if not returns_json:
        return "No return value specified"

    # Handle string returns
    if isinstance(returns_json, str):
        try:
            import json

            returns_json = json.loads(returns_json)
        except (json.JSONDecodeError, ValueError):
            return returns_json

    # Format based on structure
    if isinstance(returns_json, dict):
        lines = []

        ret_type = returns_json.get("type", "")
        description = returns_json.get("description", "")
        fields = returns_json.get("fields", [])

        if ret_type:
            lines.append(f"**Type:** {ret_type}")

        if description:
            lines.append(f"**Description:** {description}")

        if fields:
            lines.append("**Fields:**")
            for field in fields:
                if isinstance(field, dict):
                    field_name = field.get("name", "")
                    field_desc = field.get("description", "")
                    lines.append(f"  - `{field_name}`: {field_desc}")
                else:
                    lines.append(f"  - {field}")

        return "\n".join(lines) if lines else "Return value not specified"

    return str(returns_json)


def format_examples_comprehensive(examples_json: Any) -> List[str]:
    """
    Format usage examples comprehensively.

    Args:
        examples_json: Examples in JSON format

    Returns:
        List of formatted example strings
    """
    if not examples_json:
        return []

    # Handle string examples
    if isinstance(examples_json, str):
        try:
            import json

            examples_json = json.loads(examples_json)
        except (json.JSONDecodeError, ValueError):
            return [examples_json] if examples_json else []

    formatted_examples = []

    if isinstance(examples_json, list):
        for i, example in enumerate(examples_json, 1):
            if isinstance(example, dict):
                code = example.get("code", "")
                description = example.get("description", "")
                language = example.get("language", "matlab")

                example_str = f"**Example {i}**"
                if description:
                    example_str += f": {description}"
                example_str += f"\n```{language}\n{code}\n```"

                formatted_examples.append(example_str)
            else:
                formatted_examples.append(f"**Example {i}**\n```matlab\n{example}\n```")

    return formatted_examples
