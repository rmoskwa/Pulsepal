# PRP: Implement Source-Aware RAG Pipeline for PulsePal

## Context and Objective

PulsePal currently has a RAG system that doesn't differentiate between data sources in Supabase, leading to situations where users ask for function parameters but receive code examples instead. We need to implement a **source-aware RAG pipeline** that:

1. Understands the characteristics of different data sources (API docs, code examples, tutorials)
2. Routes queries to appropriate sources based on user intent
3. Formats returned data optimally for Gemini 2.5 Flash to synthesize
4. Handles multi-chunk documents coherently

## Current System Architecture

- **Base LLM**: Gemini 2.5 Flash
- **Database**: Supabase (project ID: mnbvsrsivuuuwbtkmumt)
- **Main RAG Service**: `pulsepal/rag_service.py`
- **Existing Search Methods**: Currently treats all sources equally with 50/50 split

## Implementation Requirements

### 1. Create Source Profile Configuration File

Create a new file: `pulsepal/source_profiles.py`

This file should contain:

```python
"""
Source-aware profiles for PulsePal's RAG system.
Defines characteristics, routing rules, and retrieval strategies for each data source.
"""

SOURCE_PROFILES = {
    "api_reference": {
        # Core Information
        "table_name": "api_reference",
        "search_methods": [
            "match_api_reference_search",  # RPC function
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
        
        # Query Routing
        "best_for": [
            "function parameters and types",
            "function signatures",
            "default values",
            "return types",
            "calling patterns",
            "required vs optional arguments",
            "API specifications"
        ],
        "query_indicators": [
            "parameters", "arguments", "signature", "syntax",
            "how to call", "what does X take", "inputs", "outputs",
            "returns", "default value", "required", "optional"
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
        
        # Query Routing
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
        "query_indicators": [
            "example", "show me", "implement", "code for",
            "how do I", "tutorial", "guide", "workflow",
            "integrate", "use in practice", "real code", "snippet"
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
        
        # Query Routing - EXPANDED
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
        "query_indicators": [
            # Sequence types
            "sequence", "EPI", "MPRAGE", "spin echo", "gradient echo",
            "TSE", "diffusion", "spiral", "radial",
            # Tutorial indicators
            "demo", "demonstration", "tutorial", "example sequence",
            "how to build", "learn", "educational", "teach me",
            "show me how", "walkthrough", "step by step",
            # Complete implementations
            "complete sequence", "full implementation", "template"
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

# Query Routing Rules
query_routing_rules = {
    "tutorial_detection": {
        "indicators": ["learn", "teach", "demo", "example", "how to", "walkthrough"],
        "primary_source": "official_sequence_examples",
        "secondary_source": "crawled_pages"
    },
    
    "multi_chunk_awareness": {
        "detected_documents": [
            "specification.pdf",
            "writeEpiRS.html", 
            "writeTrufi.html",
            "writeTSE.html",
            "writeHASTE.html"
        ],
        "action": "retrieve_all_related_chunks",
        "reason": "Maintain document coherence and context"
    },
    
    "source_combination_strategies": {
        "learning_query": {
            "pattern": "User wants to learn a technique",
            "sources": ["official_sequence_examples", "api_reference"],
            "order": "examples_first_then_api"
        },
        "debugging_query": {
            "pattern": "User has implementation issue",
            "sources": ["api_reference", "crawled_pages"],
            "order": "api_first_then_examples"
        }
    }
}

def get_source_indicators(query: str) -> Dict[str, float]:
    """
    Analyze query to determine which sources are most relevant.
    Returns confidence scores for each source.
    """
    # Implement indicator matching logic
```

### 2. Create Data Formatting Module

Create a new file: `pulsepal/rag_formatters.py`

1. API Reference Formatting
```python
def format_api_reference(results):
    """
    Format API reference data for optimal LLM consumption.
    Emphasizes conceptual understanding followed by technical details.
    """
    formatted_results = []
    
    for result in results:
        formatted_entry = {
            "source_type": "API_DOCUMENTATION",
            "relevance_score": result.get('similarity', 0),
            
            # Primary Information (what users usually need)
            "function": {
                "name": result['name'],
                "purpose": result['description'],  # Lead with WHAT it does
                "usage": result['calling_pattern'],  # HOW to call it correctly
            },
            
            # Technical Specifications
            "technical_details": {
                "signature": result.get('signature', ''),
                "parameters": format_parameters(result.get('parameters', {})),
                "returns": format_returns(result.get('returns', {})),
                "function_type": result.get('function_type', 'main'),
            },
            
            # Practical Context
            "examples": format_examples(result.get('usage_examples', [])),
            "related_functions": result.get('related_functions', []),
            
            # Metadata for filtering/context
            "metadata": {
                "language": result['language'],
                "has_examples": bool(result.get('usage_examples')),
                "last_updated": result.get('last_updated'),
                "source_id": result.get('source_id'),
            }
        }
        formatted_results.append(formatted_entry)
    
    return formatted_results

def format_parameters(params_json):
    """Format parameters for clear presentation."""
    if not params_json:
        return "No parameters"
    
    formatted_params = []
    for param_name, param_info in params_json.items():
        param_str = f"• {param_name}"
        if param_info.get('type'):
            param_str += f" ({param_info['type']})"
        if param_info.get('required'):
            param_str += " [REQUIRED]"
        else:
            param_str += " [optional]"
        if param_info.get('default'):
            param_str += f" = {param_info['default']}"
        if param_info.get('description'):
            param_str += f"\n    {param_info['description']}"
        formatted_params.append(param_str)
    
    return "\n".join(formatted_params)

def format_returns(returns_json):
    """Format return values for clarity."""
    if not returns_json:
        return "No return value"
    
    if isinstance(returns_json, dict):
        return f"{returns_json.get('type', 'unknown')} - {returns_json.get('description', '')}"
    return str(returns_json)

def format_examples(examples_json):
    """Format usage examples with language tags."""
    if not examples_json:
        return []
    
    formatted_examples = []
    for example in examples_json:
        formatted_examples.append({
            "code": example.get('code', ''),
            "description": example.get('description', ''),
            "language": example.get('language', 'matlab')
        })
    return formatted_examples
```
```python
def format_crawled_pages(results):
    """
    Format crawled pages with special handling for multi-chunk documents.
    Groups chunks from the same document together.
    """
    # First, group results by URL to handle multi-chunk documents
    documents_by_url = {}
    for result in results:
        url = result['url']
        if url not in documents_by_url:
            documents_by_url[url] = []
        documents_by_url[url].append(result)
    
    formatted_results = []
    
    for url, chunks in documents_by_url.items():
        # Sort chunks by chunk_number to maintain document order
        chunks.sort(key=lambda x: x.get('chunk_number', 0))
        
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

def format_multi_chunk_document(url, chunks):
    """Format multi-chunk documents with context preservation."""
    total_chunks = len(chunks)
    
    # Combine content with chunk markers
    combined_content = []
    for chunk in chunks:
        chunk_num = chunk.get('chunk_number', 0)
        content = chunk.get('content', '')
        
        if total_chunks > 1:
            combined_content.append(f"[Part {chunk_num + 1}/{total_chunks}]")
        combined_content.append(content)
    
    # Extract common metadata from first chunk
    first_chunk = chunks[0]
    metadata = first_chunk.get('metadata', {})
    
    return {
        "source_type": "DOCUMENTATION_MULTI_PART",
        "relevance_score": max(chunk.get('similarity', 0) for chunk in chunks),
        
        "document": {
            "url": url,
            "title": extract_title_from_url(url),
            "total_parts": total_chunks,
            "content": "\n\n".join(combined_content),
            "content_length": sum(len(chunk.get('content', '')) for chunk in chunks),
        },
        
        "context": {
            "source": metadata.get('source', ''),
            "file_path": metadata.get('file_path', ''),
            "language": metadata.get('language', ''),
            "document_type": get_document_type(url, metadata),
        },
        
        "metadata": {
            "source_id": first_chunk.get('source_id'),
            "repo_name": metadata.get('repo_name', ''),
            "functions_mentioned": metadata.get('all_functions', []),
            "dependencies": metadata.get('dependencies', []),
            "is_official": 'pulseq.github.io' in url,
        },
        
        "retrieval_note": f"Retrieved all {total_chunks} parts for document coherence"
    }

def format_single_chunk(chunk):
    """Format single chunk documents."""
    metadata = chunk.get('metadata', {})
    
    return {
        "source_type": "CODE_EXAMPLE" if is_code(chunk) else "DOCUMENTATION",
        "relevance_score": chunk.get('similarity', 0),
        
        "content": {
            "text": chunk.get('content', ''),
            "url": chunk.get('url', ''),
            "title": extract_title_from_url(chunk.get('url', '')),
        },
        
        "context": {
            "source": metadata.get('source', ''),
            "file_path": metadata.get('file_path', ''),
            "language": metadata.get('language', ''),
            "file_type": metadata.get('file_extension', ''),
        },
        
        "code_info": {
            "functions_used": metadata.get('all_functions', []),
            "dependencies": metadata.get('dependencies', []),
            "is_complete": not metadata.get('truncated', False),
        },
        
        "metadata": {
            "source_id": chunk.get('source_id'),
            "repo_name": metadata.get('repo_name', ''),
            "word_count": metadata.get('word_count', 0),
            "is_official": 'pulseq.github.io' in chunk.get('url', ''),
        }
    }

def is_code(chunk):
    """Determine if chunk is primarily code."""
    metadata = chunk.get('metadata', {})
    return (
        metadata.get('file_extension') in ['.m', '.py', '.cpp'] or
        metadata.get('file_category') == 'repository_file'
    )

def extract_title_from_url(url):
    """Extract a readable title from URL."""
    if not url:
        return "Untitled"
    
    # Special handling for known patterns
    if 'specification.pdf' in url:
        return "Pulseq Specification Document"
    
    # Extract filename or last path segment
    import os
    from urllib.parse import urlparse
    
    parsed = urlparse(url)
    path = parsed.path
    
    if path:
        filename = os.path.basename(path)
        # Remove extension and convert to title case
        title = os.path.splitext(filename)[0]
        return title.replace('_', ' ').replace('-', ' ').title()
    
    return parsed.netloc
```

2. Crawled Pages Formatting (with Multi-Chunk Handling)
```python
def format_official_sequence_examples(results):
    """
    Format official sequence examples as educational tutorials.
    Emphasizes learning value and practical implementation.
    """
    formatted_results = []
    
    for result in results:
        # Parse the AI summary for key insights
        summary_sections = parse_ai_summary(result.get('ai_summary', ''))
        
        formatted_entry = {
            "source_type": "EDUCATIONAL_SEQUENCE",
            "relevance_score": result.get('similarity', 0),
            
            # Educational Context (PRIMARY)
            "tutorial_info": {
                "title": extract_sequence_title(result['file_name']),
                "sequence_type": result.get('sequence_type', 'Unknown'),
                "complexity": determine_complexity(result),
                "summary": summary_sections.get('overview', ''),
                "key_techniques": summary_sections.get('techniques', []),
                "learning_points": summary_sections.get('learning_points', []),
            },
            
            # Complete Implementation
            "implementation": {
                "full_code": result.get('content', ''),
                "code_length": len(result.get('content', '')),
                "language": "MATLAB",  # All official examples are MATLAB
            },
            
            # Technical Specifications
            "technical_details": {
                "trajectory_type": result.get('trajectory_type', ''),
                "acceleration": result.get('acceleration', 'none'),
                "performance_notes": summary_sections.get('performance', ''),
                "warnings": summary_sections.get('warnings', []),
            },
            
            # Usage Context
            "usage_guide": {
                "when_to_use": generate_usage_context(result),
                "prerequisites": extract_prerequisites(result),
                "modifications_suggested": summary_sections.get('customization', []),
            },
            
            # Metadata
            "metadata": {
                "file_name": result['file_name'],
                "is_demo": 'demo' in result['file_name'].lower(),
                "validation_status": "official_tested",
                "suitable_for_learning": True,
            }
        }
        formatted_results.append(formatted_entry)
    
    return formatted_results

def parse_ai_summary(summary):
    """Parse AI summary into structured sections."""
    sections = {
        'overview': '',
        'techniques': [],
        'learning_points': [],
        'performance': '',
        'warnings': [],
        'customization': []
    }
    
    if not summary:
        return sections
    
    # Extract overview (usually first paragraph)
    paragraphs = summary.split('\n\n')
    if paragraphs:
        sections['overview'] = paragraphs[0]
    
    # Look for key phrases
    summary_lower = summary.lower()
    
    # Extract techniques mentioned
    if 'technique' in summary_lower or 'method' in summary_lower:
        # Parse for technical terms
        techniques = extract_technical_terms(summary)
        sections['techniques'] = techniques
    
    # Extract warnings
    if 'warning' in summary_lower or 'caution' in summary_lower:
        sections['warnings'].append("Contains important warnings - review carefully")
    
    # Extract performance notes
    if 'performance' in summary_lower or 'optimiz' in summary_lower:
        sections['performance'] = "Includes performance optimizations"
    
    return sections

def determine_complexity(result):
    """Determine complexity level based on various factors."""
    content_length = len(result.get('content', ''))
    sequence_type = result.get('sequence_type', '')
    
    # Simple heuristic based on length and type
    if content_length < 7000:
        return "beginner"
    elif content_length < 15000:
        return "intermediate"
    else:
        return "advanced"
    
    # Adjust for specific types
    if sequence_type in ['Diffusion', 'EPI']:
        return "advanced"
    
    return "intermediate"

def extract_sequence_title(file_name):
    """Extract readable title from filename."""
    import os
    
    # Remove path and extension
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    
    # Remove common prefixes
    for prefix in ['write', 'demo', 'test']:
        if base_name.lower().startswith(prefix):
            base_name = base_name[len(prefix):]
    
    # Convert to readable format
    # writeEpiDiffusionRS_PMC -> EPI Diffusion RS PMC
    import re
    title = re.sub(r'([a-z])([A-Z])', r'\1 \2', base_name)
    title = title.replace('_', ' - ')
    
    return title.strip()
```  

3. Official Sequence Examples Formatting
```python
def format_unified_response(source_results, query_context):
    """
    Create a unified response structure that Gemini can easily process.
    Handles mixed sources and provides clear context.
    """
    response = {
        "query": query_context.get('original_query', ''),
        "search_metadata": {
            "sources_searched": list(source_results.keys()),
            "total_results": sum(len(results) for results in source_results.values()),
            "timestamp": datetime.now().isoformat(),
        },
        
        "results_by_source": {},
        "synthesis_hints": [],
        "citation_map": {}
    }
    
    # Process each source type
    for source_type, results in source_results.items():
        if source_type == "api_reference":
            formatted = format_api_reference(results)
            response["results_by_source"]["api_documentation"] = formatted
            response["synthesis_hints"].append(
                "API documentation provides authoritative function specifications"
            )
            
        elif source_type == "crawled_pages":
            formatted = format_crawled_pages(results)
            response["results_by_source"]["examples_and_docs"] = formatted
            response["synthesis_hints"].append(
                "Examples show practical implementation patterns"
            )
            
        elif source_type == "official_sequence_examples":
            formatted = format_official_sequence_examples(results)
            response["results_by_source"]["tutorials"] = formatted
            response["synthesis_hints"].append(
                "Official sequences are tested and educational"
            )
    
    # Add citation guidance
    response["citation_map"] = generate_citation_map(response["results_by_source"])
    
    # Add synthesis recommendations
    response["synthesis_recommendations"] = generate_synthesis_recommendations(
        source_results, 
        query_context
    )
    
    return response

def generate_citation_map(results_by_source):
    """Create a map for proper citation of sources."""
    citation_map = {}
    citation_id = 1
    
    for source_type, results in results_by_source.items():
        for result in results:
            if source_type == "api_documentation":
                citation_key = f"[{citation_id}]"
                citation_text = f"{result['function']['name']} - Pulseq API Documentation"
                
            elif source_type == "examples_and_docs":
                citation_key = f"[{citation_id}]"
                doc_title = result.get('document', result.get('content', {})).get('title', 'Unknown')
                citation_text = f"{doc_title}"
                
            elif source_type == "tutorials":
                citation_key = f"[{citation_id}]"
                citation_text = f"{result['tutorial_info']['title']} - Official Example"
            
            citation_map[citation_key] = citation_text
            citation_id += 1
    
    return citation_map

def generate_synthesis_recommendations(source_results, query_context):
    """Generate recommendations for how to synthesize the results."""
    recommendations = []
    
    # Check what types of results we have
    has_api = "api_reference" in source_results and source_results["api_reference"]
    has_examples = "crawled_pages" in source_results and source_results["crawled_pages"]
    has_tutorials = "official_sequence_examples" in source_results and source_results["official_sequence_examples"]
    
    # Query type detection
    query_lower = query_context.get('original_query', '').lower()
    is_learning = any(word in query_lower for word in ['learn', 'tutorial', 'how to', 'teach'])
    is_parameter = any(word in query_lower for word in ['parameter', 'argument', 'signature'])
    is_debug = any(word in query_lower for word in ['error', 'wrong', 'fix', 'debug'])
    
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
    
    return recommendations
```


### 3. Modify RAG Service

Update `pulsepal/rag_service.py`:

#### 3.1 Add Source-Aware Search Method

```python
async def search_with_source_awareness(
    self,
    query: str,
    sources: Optional[List[str]] = None,
    forced: bool = False,
    source_hints: Optional[Dict] = None
) -> Dict:
    """
    Perform source-aware search across Supabase tables.
    
    Args:
        query: User's search query
        sources: Specific sources to search (if None, auto-determine)
        forced: Whether this search was forced by semantic routing
        source_hints: Additional hints about which sources to prioritize
    
    Returns:
        Formatted results organized by source with synthesis hints
    """
    # Implementation steps:
    # 1. Determine which sources to search if not specified
    # 2. Search each source with appropriate methods
    # 3. Handle multi-chunk documents
    # 4. Format results using rag_formatters
    # 5. Return unified response
```

#### 3.2 Implement Source-Specific Search Methods

```python
async def _search_api_reference(self, query: str, limit: int = 5) -> List[Dict]:
    """Search API reference with higher precision threshold."""
    # Use match_api_reference_search RPC
    # Apply language filtering if needed
    # Return structured results

async def _search_crawled_pages(self, query: str, limit: int = 10) -> List[Dict]:
    """Search crawled pages with chunk coherence."""
    # Use match_crawled_pages RPC
    # Group multi-chunk documents
    # Return complete documents

async def _search_official_sequences(self, query: str, limit: int = 3) -> List[Dict]:
    """Search official sequences for tutorials."""
    # Use match_official_sequences RPC
    # Filter by sequence type if query specifies
    # Return educational sequences
```

#### 3.3 Add Multi-Chunk Document Handler

```python
async def _retrieve_all_chunks(self, url: str) -> List[Dict]:
    """
    Retrieve all chunks for a multi-chunk document.
    Critical for documents like specification.pdf (9 chunks).
    """
    # Query all chunks with same URL
    # Sort by chunk_number
    # Return ordered list
```

### 4. Update Tool Integration

Modify `pulsepal/tools.py` to use source-aware search:

```python
@pulsepal_agent.tool
async def search_pulseq_knowledge(
    ctx,
    query: str,
    limit: int = 30,
    sources: Optional[List[str]] = None,
    forced: bool = False
) -> str:
    """
    Enhanced search with source awareness.
    """
    # Get source hints from semantic router if available
    source_hints = ctx.deps.get('forced_search_hints', {})
    
    # Use new source-aware search
    results = await ctx.deps.rag_service.search_with_source_awareness(
        query=query,
        sources=sources,
        forced=forced,
        source_hints=source_hints
    )
    
    # Return formatted results
    return results
```

### 5. Update System Prompt

Modify the system prompt in `pulsepal/main_agent.py` to include source awareness:

```python
SYSTEM_PROMPT_ADDITION = """
## Source-Aware Information Retrieval

When searching for information, you have access to three specialized sources:

1. **API_REFERENCE**: Structured documentation with function signatures, parameters, and returns
   - Use for: Parameter questions, function specifications, calling patterns
   - Characteristics: Authoritative, complete parameter documentation

2. **CRAWLED_PAGES**: Code examples and tutorials from GitHub and documentation
   - Use for: Implementation examples, how-to guides, troubleshooting
   - Note: Some documents span multiple chunks - system retrieves all parts automatically

3. **OFFICIAL_SEQUENCES**: Complete, tested MRI sequence tutorials
   - Use for: Learning sequence construction, starting templates, best practices
   - Characteristics: Educational, validated, includes detailed explanations

The system automatically determines which sources to search based on your query.
Results include synthesis hints to help you combine information effectively.
"""
```

### 6. Testing Requirements

Create `tests/test_source_aware_rag.py`:

```python
"""Test source-aware RAG functionality."""

def test_api_parameter_query():
    """Ensure parameter queries prioritize API reference."""
    # Query: "What are the parameters of makeTrapezoid?"
    # Should return api_reference results first

def test_tutorial_query():
    """Ensure tutorial queries return official sequences."""
    # Query: "Show me an EPI sequence example"
    # Should return official_sequence_examples

def test_multi_chunk_handling():
    """Ensure multi-chunk documents are retrieved completely."""
    # Query about specification.pdf content
    # Should retrieve all 9 chunks

def test_source_fallback():
    """Test fallback when primary source has low confidence."""
    # Query with low confidence in primary source
    # Should search secondary sources

def test_mixed_source_query():
    """Test queries that benefit from multiple sources."""
    # Query: "How do I use makeTrapezoid with proper parameters?"
    # Should return both API docs and examples
```

## Implementation Steps

1. **Phase 1: Core Infrastructure**
   - Create `source_profiles.py` with complete profiles
   - Create `rag_formatters.py` with all formatting functions
   - Add tests for formatting functions

2. **Phase 2: RAG Service Enhancement**
   - Implement `search_with_source_awareness()` method
   - Add source-specific search methods
   - Implement multi-chunk document handling
   - Test with real Supabase queries

3. **Phase 3: Integration**
   - Update `tools.py` to use new search method
   - Update system prompt with source descriptions
   - Ensure backward compatibility

4. **Phase 4: Testing & Validation**
   - Run comprehensive tests
   - Test with actual user queries from conversation logs
   - Verify improved accuracy for parameter questions

## Key Implementation Details

### Multi-Chunk Document Handling
```python
# Documents that require special handling:
MULTI_CHUNK_DOCUMENTS = [
    "specification.pdf",  # 9 chunks
    "writeEpiRS.html",    # 3 chunks
    "writeTrufi.html",    # 3 chunks
    "writeTSE.html",      # 3 chunks
    "writeHASTE.html"     # 3 chunks
]
```

### Source Priority Logic
```python
# Example routing logic:
if "parameters" in query or "arguments" in query:
    primary_source = "api_reference"
elif "example" in query or "show me" in query:
    primary_source = "crawled_pages"
elif "tutorial" in query or "learn" in query:
    primary_source = "official_sequence_examples"
```

### Performance Considerations
- Cache source profiles at startup
- Use parallel searches only when multiple sources needed
- Implement early termination if high-confidence match found
- Consider caching strategies based on source stability

## Success Criteria

1. **Accuracy**: Parameter questions return API documentation, not just code examples
2. **Coherence**: Multi-chunk documents are retrieved completely
3. **Performance**: Response time remains under 2 seconds for typical queries
4. **Backward Compatibility**: Existing functionality remains intact
5. **Educational Value**: Tutorial queries return official sequences with explanations

## Error Handling

- Gracefully handle when a source returns no results
- Implement fallback to broader search if specific source fails
- Log source routing decisions for debugging
- Handle Supabase connection issues gracefully

## Configuration

Add to `.env` or configuration:
```python
# Source-aware RAG configuration
RAG_API_THRESHOLD = 0.75  # Higher threshold for API docs
RAG_EXAMPLE_THRESHOLD = 0.7  # Standard threshold for examples
RAG_TUTORIAL_THRESHOLD = 0.65  # Lower threshold for tutorials
RAG_MAX_CHUNKS_PER_DOC = 10  # Maximum chunks to retrieve per document
```

## Notes for Implementation

1. **Preserve existing `search_pulseq_knowledge` functionality** - add source awareness as enhancement
2. **Source profiles should be configurable** - not hard-coded in case we need to adjust
3. **Formatting should preserve all metadata** - Gemini might need it for decisions
4. **Consider logging source routing decisions** - helpful for debugging and optimization
5. **Multi-chunk handling is critical** - specification.pdf is important documentation

## Example Usage After Implementation

```python
# User asks: "What are the parameters of mr.makeTrapezoid?"

# System internally:
# 1. Identifies this as API documentation query
# 2. Searches api_reference with high threshold
# 3. Formats results with description, parameters, examples
# 4. Returns structured response

# Gemini receives:
{
    "source_type": "API_DOCUMENTATION",
    "function": {
        "name": "makeTrapezoid",
        "purpose": "Generates a trapezoidal gradient waveform...",
        "usage": "mr.makeTrapezoid(channel, ...)"
    },
    "technical_details": {
        "parameters": "• channel (string) [REQUIRED]...",
        # ... complete parameter list
    }
}

# Result: User gets accurate parameter information, not just code examples
```

This implementation will make PulsePal's RAG system more intelligent and accurate, especially for function-specific queries that currently return suboptimal results.