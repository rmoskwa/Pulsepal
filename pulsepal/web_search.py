"""
Web search functionality for Pulsepal.

This module provides web search capabilities for finding MRI-related information
from external sources when not available in the RAG database.
"""

import logging
from typing import List, Dict, Any, Optional
import os

logger = logging.getLogger(__name__)


class WebSearchService:
    """Service for performing web searches for MRI information."""
    
    def __init__(self):
        """Initialize web search service."""
        # Note: In a production environment, you would configure a search API here
        # For now, we'll provide a simple interface that can be extended
        logger.info("Web search service initialized")
    
    def search_mri_information(
        self,
        query: str,
        max_results: int = 5,
        focus_areas: Optional[List[str]] = None
    ) -> str:
        """
        Search the web for MRI-related information.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            focus_areas: Optional list of areas to focus on (e.g., ["physics", "sequences", "safety"])
            
        Returns:
            Formatted search results or informative message
        """
        try:
            # Add MRI context to the query
            mri_query = f"MRI {query}"
            if focus_areas:
                mri_query += f" {' '.join(focus_areas)}"
            
            # For now, return an informative message about what would be searched
            # In production, this would call an actual search API
            results = self._mock_search_results(mri_query, max_results)
            
            return self._format_web_results(results, query)
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return f"Error performing web search: {str(e)}"
    
    def search_pulseq_resources(
        self,
        query: str,
        resource_type: Optional[str] = None
    ) -> str:
        """
        Search for Pulseq-specific resources on the web.
        
        Args:
            query: Search query
            resource_type: Type of resource (e.g., "tutorial", "example", "paper")
            
        Returns:
            Formatted search results
        """
        try:
            # Construct Pulseq-specific query
            pulseq_query = f"Pulseq {query}"
            if resource_type:
                pulseq_query += f" {resource_type}"
            
            # Mock search results for now
            results = self._mock_pulseq_resources(pulseq_query, resource_type)
            
            return self._format_pulseq_results(results, query)
            
        except Exception as e:
            logger.error(f"Pulseq resource search failed: {e}")
            return f"Error searching Pulseq resources: {str(e)}"
    
    def _mock_search_results(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Mock search results for demonstration.
        In production, this would call an actual search API.
        """
        # Provide some example results that would typically come from a search API
        mock_results = [
            {
                "title": "MRI Physics: Basic Principles",
                "url": "https://www.mriquestions.com/index.html",
                "snippet": "Comprehensive resource for MRI physics questions and answers...",
                "source": "MRI Questions"
            },
            {
                "title": "Pulseq: Open Source MRI Sequences",
                "url": "https://pulseq.github.io/",
                "snippet": "Official Pulseq documentation and tutorials for sequence programming...",
                "source": "Pulseq Official"
            },
            {
                "title": "MRI Sequence Design Principles",
                "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3998685/",
                "snippet": "Scientific article on MRI pulse sequence design and optimization...",
                "source": "PubMed Central"
            }
        ]
        
        return mock_results[:max_results]
    
    def _mock_pulseq_resources(
        self, 
        query: str, 
        resource_type: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Mock Pulseq-specific resources."""
        resources = [
            {
                "title": "Pulseq GitHub Repository",
                "url": "https://github.com/pulseq/pulseq",
                "snippet": "Main Pulseq repository with MATLAB/Octave implementation and examples",
                "type": "repository"
            },
            {
                "title": "PyPulseq Documentation",
                "url": "https://pypulseq.readthedocs.io/",
                "snippet": "Python implementation of Pulseq with API documentation",
                "type": "documentation"
            },
            {
                "title": "Pulseq Tutorials",
                "url": "https://github.com/pulseq/tutorials",
                "snippet": "Step-by-step tutorials for creating MRI sequences with Pulseq",
                "type": "tutorial"
            }
        ]
        
        if resource_type:
            resources = [r for r in resources if r.get("type") == resource_type]
        
        return resources
    
    def _format_web_results(
        self, 
        results: List[Dict[str, Any]], 
        query: str
    ) -> str:
        """Format web search results for display."""
        if not results:
            return f"No web results found for query: '{query}'"
        
        formatted = [f"## Web Search Results for: '{query}'\n"]
        formatted.append(
            "*Note: Web search functionality is currently in demonstration mode. "
            "In production, this would search actual web resources.*\n"
        )
        
        for i, result in enumerate(results, 1):
            formatted.append(f"### {i}. {result['title']}")
            formatted.append(f"**Source:** {result['source']}")
            formatted.append(f"**URL:** {result['url']}")
            formatted.append(f"{result['snippet']}")
            formatted.append("")
        
        formatted.append(
            "\n*For immediate assistance, please check the RAG database using "
            "the documentation search tools, or visit the official Pulseq website.*"
        )
        
        return "\n".join(formatted)
    
    def _format_pulseq_results(
        self, 
        results: List[Dict[str, Any]], 
        query: str
    ) -> str:
        """Format Pulseq resource results."""
        if not results:
            return f"No Pulseq resources found for query: '{query}'"
        
        formatted = [f"## Pulseq Resources for: '{query}'\n"]
        
        # Group by type
        by_type = {}
        for result in results:
            res_type = result.get("type", "other")
            if res_type not in by_type:
                by_type[res_type] = []
            by_type[res_type].append(result)
        
        # Format grouped results
        type_names = {
            "repository": "Code Repositories",
            "documentation": "Documentation",
            "tutorial": "Tutorials",
            "paper": "Research Papers",
            "other": "Other Resources"
        }
        
        for res_type, items in by_type.items():
            formatted.append(f"### {type_names.get(res_type, res_type.title())}")
            for item in items:
                formatted.append(f"- **{item['title']}**")
                formatted.append(f"  {item['snippet']}")
                formatted.append(f"  URL: {item['url']}")
            formatted.append("")
        
        return "\n".join(formatted)


# Global instance
_web_search_service: Optional[WebSearchService] = None


def get_web_search_service() -> WebSearchService:
    """
    Get the global web search service instance.
    
    Returns:
        WebSearchService instance
    """
    global _web_search_service
    if _web_search_service is None:
        _web_search_service = WebSearchService()
    return _web_search_service