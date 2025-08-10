"""
Source-aware profiles for PulsePal's RAG system.
Defines characteristics, routing rules, and retrieval strategies for each data source.
"""

from typing import Dict, Any

SOURCE_PROFILES = {
    "api_reference": {
        # Core Information
        "table_name": "api_reference",
        "search_methods": [
            "match_api_reference",  # RPC function
            "api_reference_search",         # View for MATLAB
            "api_reference_details",        # Detailed view
            "function_calling_patterns"     # Usage patterns view
        ],
        "summary": "Structured API documentation with function signatures, parameters, return values, and calling patterns",
        
        # Enhanced Metadata Schema
        "metadata_schema": {
            "function_name": "string",
            "language": "enum[matlab,python,cpp]",
            "signature": "text",  # Full function signature
            "description": "text",  # IMPORTANT: Conceptual explanation (avg 329 chars)
            "parameters": "jsonb",  # Detailed parameter specs
            "returns": "jsonb",  # Return value documentation
            "function_type": "enum[main,helper,internal,class]",
            "usage_examples": "jsonb",
            "calling_pattern": "string",
            "related_functions": "array[string]",
            "has_nargin_pattern": "boolean",
            "last_updated": "timestamp"
        },
        
        # Best Use Cases (for LLM understanding)
        "best_for": [
            "function parameters and types",
            "function signatures",
            "default values",
            "return types",
            "calling patterns",
            "required vs optional arguments",
            "API specifications"
        ],
        
        # Data Characteristics
        "data_characteristics": {
            "total_records": 472,
            "languages": {"matlab": 150, "python": 282, "cpp": 40},
            "structured_format": "JSON parameters and returns",
            "has_usage_examples": True,
            "freshness": "stable - functions rarely change"
        },
        
        # Retrieval Configuration
        "retrieval_config": {
            "similarity_threshold": 0.75,  # Higher threshold for precision
            "max_results": 5,
            "ranking_priority": "exact_match",
            "filter_strategy": "language_specific"
        },
        
        # Quality Indicators
        "quality_metrics": {
            "completeness": "high",  # All params documented
            "accuracy": "verified",   # From official sources
            "coverage": "comprehensive"
        },
        
        # Return Format
        "return_format": {
            "type": "structured_api_doc",
            "include_fields": [
                "name",
                "description",  # ALWAYS include this!
                "parameters",
                "calling_pattern",
                "returns",
                "usage_examples"
            ],
            "presentation_order": [
                "description",  # Start with WHAT it does
                "calling_pattern",  # Show HOW to call it
                "parameters",  # Detail the inputs
                "returns",  # What you get back
                "usage_examples"  # Practical examples
            ],
            "citation_style": "function_reference"
        }
    },
    
    "crawled_pages": {
        # Core Information
        "table_name": "crawled_pages",
        "search_methods": [
            "match_crawled_pages",    # General search
            "match_code_examples"      # Code-specific search
        ],
        "summary": "Code examples, tutorials, documentation chunks from GitHub repositories and documentation sites",
        
        # Enhanced Metadata Schema - KEEPING chunk_number!
        "metadata_schema": {
            "source_id": "string",
            "url": "string",
            "chunk_number": "integer",  # IMPORTANT: Tracks multi-part documents
            "language": "enum[matlab,python,cpp]",
            "file_path": "string",
            "repo_name": "string",
            "source_type": "enum[github,documentation]",
            "file_category": "enum[repository_file,documentation]",
            "dependencies": "array[string]",
            "all_functions": "array[string]",
            "word_count": "integer"
        },
        
        # Best Use Cases (for LLM understanding)
        "best_for": [
            "implementation examples",
            "complete code snippets",
            "how-to guides",
            "specification details",  # The PDF spec is here!
            "troubleshooting",
            "real-world usage",
            "integration patterns",
            "workflow examples"
        ],
        
        # Chunk Handling Strategy
        "chunk_handling": {
            "multi_chunk_documents": [
                "specification.pdf (9 chunks)",
                "writeEpiRS.html (3 chunks)",
                "writeTrufi.html (3 chunks)",
                "writeTSE.html (3 chunks)",
                "writeHASTE.html (3 chunks)"
            ],
            "retrieval_strategy": "retrieve_all_chunks_for_context",
            "importance": "Critical for maintaining document coherence"
        },
        
        # Data Characteristics
        "data_characteristics": {
            "total_records": 1703,
            "avg_chunk_size": "~5000 chars",
            "source_diversity": 27,
            "content_type": "mixed (code + prose)",
            "freshness": "varies - some repos updated frequently"
        },
        
        # Retrieval Configuration
        "retrieval_config": {
            "similarity_threshold": 0.7,
            "max_results": 10,
            "ranking_priority": "semantic_similarity",
            "chunk_strategy": "include_surrounding_context"
        },
        
        # Quality Indicators
        "quality_metrics": {
            "source_credibility": "variable",  # Mix of official and community
            "recency": "mixed",
            "practical_value": "high"
        },
        
        # Return Format
        "return_format": {
            "type": "code_with_context",
            "include_fields": ["content", "url", "source_id", "metadata"],
            "citation_style": "source_attribution",
            "context_window": 2  # Include neighboring chunks if relevant
        }
    },
    
    "official_sequence_examples": {
        # Core Information
        "table_name": "official_sequence_examples",  # This is a view
        "search_methods": ["match_official_sequences"],
        "summary": "Complete, tested, educational MRI sequence demonstrations with detailed explanations",
        
        # Enhanced Metadata Schema
        "metadata_schema": {
            "file_name": "string",  # Often includes 'demoSeq' path
            "sequence_type": "enum[Diffusion,EPI,MPRAGE,TSE,etc]",
            "trajectory_type": "enum[radial,spiral,EPI,3D,Cartesian]",
            "acceleration": "enum[GRAPPA,SENSE,compressed_sensing,none]",
            "ai_summary": "text",  # Detailed educational descriptions!
            "content_length": "integer",  # Ranges from 5KB to 21KB
            "educational_value": "high"  # All are demo/educational
        },
        
        # Best Use Cases (for LLM understanding)
        "best_for": [
            "complete sequence templates",
            "educational demonstrations",
            "learning sequence construction",
            "understanding Pulseq workflow",
            "reference implementations",
            "starting points for new sequences",
            "best practices examples",
            "fMRI quality control examples",
            "diffusion-weighted imaging tutorials"
        ],
        
        # Data Characteristics
        "data_characteristics": {
            "total_sequences": 39,
            "all_tested": True,
            "all_educational": True,  # Key finding!
            "includes_detailed_summaries": True,
            "code_length_range": "5KB-21KB",
            "documentation_quality": "excellent"
        },
        
        # Retrieval Configuration
        "retrieval_config": {
            "similarity_threshold": 0.65,  # Lower threshold to catch variations
            "max_results": 3,  # Fewer results - each is complete sequence
            "ranking_priority": "category_match_first",
            "filter_strategy": "sequence_type_filtering"
        },
        
        # Quality Indicators
        "quality_metrics": {
            "validation": "official",
            "completeness": "full sequences",
            "best_practices": "follows standards"
        },
        
        # Return Format
        "return_format": {
            "type": "educational_sequence",
            "include_fields": ["content", "ai_summary", "sequence_type", "file_name"],
            "highlight_educational_aspects": True,
            "presentation": "tutorial_style",
            "citation_style": "official_example_reference"
        }
    }
}

ROUTING_CONFIGURATION = {
    "fallback_strategy": {
        "primary_failure_threshold": 0.5,
        "secondary_sources": ["crawled_pages"],  # If API ref fails, try examples
        "combine_sources": True  # Can search multiple sources if needed
    },
    
    "freshness_requirements": {
        "api_reference": "stable",  # Functions rarely change
        "crawled_pages": "check_monthly",  # Some repos update frequently
        "official_sequence_examples": "stable"  # Official examples are stable
    },
    
    "performance_optimization": {
        "cache_duration": {
            "api_reference": 3600,  # 1 hour cache
            "crawled_pages": 1800,   # 30 min cache
            "official_sequence_examples": 7200  # 2 hour cache
        },
        "parallel_search": False,  # Search sources sequentially for now
        "early_termination": True  # Stop if high confidence match found
    }
}

# Multi-chunk Document Handling (still needed for document coherence)
MULTI_CHUNK_AWARENESS = {
    "detected_documents": [
        "specification.pdf",
        "writeEpiRS.html", 
        "writeTrufi.html",
        "writeTSE.html",
        "writeHASTE.html"
    ],
    "action": "retrieve_all_related_chunks",
    "reason": "Maintain document coherence and context"
}

# Multi-chunk documents that require special handling
MULTI_CHUNK_DOCUMENTS = [
    "specification.pdf",  # 9 chunks
    "writeEpiRS.html",    # 3 chunks
    "writeTrufi.html",    # 3 chunks
    "writeTSE.html",      # 3 chunks
    "writeHASTE.html"     # 3 chunks
]


# Removed deprecated keyword-based functions
# The LLM now decides which sources to search based on query understanding
# No keyword matching is performed - source selection is entirely LLM-driven


def get_source_profile(source_name: str) -> Dict[str, Any]:
    """
    Get the complete profile for a data source.
    
    Args:
        source_name: Name of the source
        
    Returns:
        Source profile dictionary
    """
    return SOURCE_PROFILES.get(source_name, {})


def get_retrieval_config(source_name: str) -> Dict[str, Any]:
    """
    Get retrieval configuration for a specific source.
    
    Args:
        source_name: Name of the source
        
    Returns:
        Retrieval configuration dictionary
    """
    profile = get_source_profile(source_name)
    return profile.get("retrieval_config", {})