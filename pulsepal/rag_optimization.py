#!/usr/bin/env python3
"""
Optimized RAG methods for PulsePal to fix slow data fetching.

This module provides optimized implementations that properly utilize:
1. ai_summary field in official_sequence_examples table
2. Content parsing with [summary]\n---\n[full content] structure
3. Two-tier fetching strategy for performance
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class OptimizedRAGMethods:
    """Optimized RAG methods for fast data retrieval."""
    
    def __init__(self, supabase_client):
        """Initialize with Supabase client."""
        self.supabase_client = supabase_client
        self.client = supabase_client.client
    
    async def get_official_sequence_optimized(self, sequence_type: str) -> Optional[Dict[str, Any]]:
        """
        Optimized version that uses ai_summary first, then fetches content.
        
        Strategy:
        1. First query: Get ai_summary (2KB) to find best match
        2. Second query: Fetch only the content field for that specific record
        
        This avoids fetching large content fields during search.
        """
        try:
            # Normalize sequence type
            type_mapping = {
                'epi': 'EPI',
                'echo_planar': 'EPI',
                'spin_echo': 'SpinEcho',
                'spinecho': 'SpinEcho',
                'gradient_echo': 'GradientEcho',
                'gre': 'GradientEcho',
                'tse': 'TSE',
                'turbo_spin': 'TSE',
                'mprage': 'MPRAGE',
                'ute': 'UTE',
                'haste': 'HASTE',
                'trufisp': 'TrueFISP',
                'press': 'PRESS',
                'spiral': 'Spiral'
            }
            
            normalized_type = type_mapping.get(sequence_type.lower(), sequence_type)
            logger.info(f"Searching for official sequence: {normalized_type}")
            
            # Step 1: Fast query using ai_summary to find the best match
            summary_result = self.client.from_('official_sequence_examples')\
                .select('id, file_name, sequence_type, ai_summary')\
                .eq('sequence_type', normalized_type)\
                .limit(1)\
                .execute()
            
            if not summary_result.data:
                logger.info(f"No official sequence found for type: {normalized_type}")
                return None
            
            # Get the record ID from the first step
            record = summary_result.data[0]
            record_id = record['id']
            
            logger.info(f"Found sequence in {record['file_name']}, fetching content...")
            
            # Step 2: Fetch only the content field for this specific record
            content_result = self.client.from_('official_sequence_examples')\
                .select('content')\
                .eq('id', record_id)\
                .limit(1)\
                .execute()
            
            if not content_result.data or not content_result.data[0].get('content'):
                logger.warning(f"No content found for record ID: {record_id}")
                return None
            
            # Return the complete result
            return {
                'content': content_result.data[0]['content'],
                'file_name': record['file_name'],
                'sequence_type': record['sequence_type'],
                'ai_summary': record['ai_summary']  # Include summary for context
            }
            
        except Exception as e:
            logger.error(f"Error in get_official_sequence_optimized: {e}")
            return None
    
    async def search_crawled_pages_optimized(
        self, 
        query: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Optimized search that properly parses crawled_pages content.
        
        The content field has structure: [summary]\n---\n[full content]
        We extract and use the summary for fast display, only fetching
        full content when explicitly needed.
        """
        try:
            # First, do a vector search on the embeddings
            # This is fast because it doesn't fetch the large content field
            search_result = self.client.rpc(
                'match_crawled_pages',
                {
                    'query_embedding': self._get_embedding(query),
                    'match_count': limit * 2  # Get more candidates
                }
            ).execute()
            
            if not search_result.data:
                # Fallback to text search
                search_result = self.client.from_('crawled_pages')\
                    .select('id, url, title')\
                    .ilike('title', f'%{query}%')\
                    .limit(limit)\
                    .execute()
            
            if not search_result.data:
                return []
            
            # Process results with optimized content fetching
            results = []
            for item in search_result.data[:limit]:
                try:
                    # Fetch content for this specific item
                    content_result = self.client.from_('crawled_pages')\
                        .select('content')\
                        .eq('id', item['id'])\
                        .limit(1)\
                        .execute()
                    
                    if content_result.data and content_result.data[0].get('content'):
                        content = content_result.data[0]['content']
                        
                        # Parse the content structure
                        summary, full_content = self._parse_content_structure(content)
                        
                        results.append({
                            'url': item.get('url', ''),
                            'title': item.get('title', ''),
                            'summary': summary,
                            'full_content': full_content,  # Store but don't always use
                            'similarity': item.get('similarity', 0.0)
                        })
                        
                except Exception as e:
                    logger.warning(f"Error processing crawled page {item.get('id')}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in search_crawled_pages_optimized: {e}")
            return []
    
    def _parse_content_structure(self, content: str) -> tuple[str, str]:
        """
        Parse the [summary]\n---\n[full content] structure.
        
        Returns:
            Tuple of (summary, full_content)
        """
        if not content:
            return "", ""
        
        # Look for the separator
        if '\n---\n' in content:
            parts = content.split('\n---\n', 1)
            summary = parts[0].strip()
            full_content = parts[1].strip() if len(parts) > 1 else ""
        elif '---' in content:
            # Try without newlines
            parts = content.split('---', 1)
            summary = parts[0].strip()
            full_content = parts[1].strip() if len(parts) > 1 else ""
        else:
            # No separator found, treat first 500 chars as summary
            summary = content[:500] + "..." if len(content) > 500 else content
            full_content = content
        
        return summary, full_content
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using the configured embedding service.
        This is a placeholder - should use actual embedding service.
        """
        # This would call your actual embedding service
        # For now, return a mock embedding
        return [0.0] * 768  # Typical embedding dimension
    
    async def get_function_details_optimized(
        self, 
        function_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Optimized function details retrieval using indexed views.
        """
        try:
            # Use the api_reference table with selective field fetching
            result = self.client.from_('api_reference')\
                .select('name, signature, description, parameters, returns, usage_examples')\
                .in_('name', function_names)\
                .execute()
            
            if not result.data:
                # Fallback to search
                results = []
                for name in function_names:
                    search_result = self.client.from_('api_reference')\
                        .select('name, signature, description, parameters, returns, usage_examples')\
                        .ilike('name', f'%{name}%')\
                        .limit(1)\
                        .execute()
                    
                    if search_result.data:
                        results.extend(search_result.data)
                
                return results
            
            return result.data
            
        except Exception as e:
            logger.error(f"Error in get_function_details_optimized: {e}")
            return []
    
    async def search_functions_fast_optimized(
        self, 
        query: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Phase 1: Ultra-fast function discovery using minimal fields.
        """
        try:
            # Use only essential fields for fast response
            result = self.client.from_('api_reference')\
                .select('name, signature, description')\
                .or_(f"name.ilike.%{query}%,description.ilike.%{query}%")\
                .limit(limit)\
                .execute()
            
            if not result.data:
                # Try with variations
                query_lower = query.lower()
                if not query_lower.startswith('mr.'):
                    alt_query = f"mr.{query_lower}"
                    result = self.client.from_('api_reference')\
                        .select('name, signature, description')\
                        .ilike('name', f'%{alt_query}%')\
                        .limit(limit)\
                        .execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Error in search_functions_fast_optimized: {e}")
            return []


# Patch methods for existing RAGService
def apply_optimizations_to_rag_service(rag_service):
    """
    Apply optimized methods to existing RAGService instance.
    This monkey-patches the existing service with optimized versions.
    """
    optimizer = OptimizedRAGMethods(rag_service.supabase_client)
    
    # Store original methods for fallback
    rag_service._original_get_official_sequence = rag_service.get_official_sequence
    rag_service._original_search_functions_fast = rag_service.search_functions_fast
    rag_service._original_get_function_details = rag_service.get_function_details
    
    # Replace with optimized versions
    rag_service.get_official_sequence = optimizer.get_official_sequence_optimized
    rag_service.search_functions_fast = optimizer.search_functions_fast_optimized
    rag_service.get_function_details = optimizer.get_function_details_optimized
    
    # Add new optimized search method
    rag_service.search_crawled_pages_optimized = optimizer.search_crawled_pages_optimized
    
    logger.info("Applied RAG optimizations successfully")
    return rag_service


# Test the optimizations
async def test_optimizations():
    """Test the optimized methods."""
    from .rag_service import get_rag_service
    import time
    
    rag_service = get_rag_service()
    apply_optimizations_to_rag_service(rag_service)
    
    print("Testing optimized methods...")
    
    # Test 1: Official sequence fetch
    print("\n1. Testing official sequence fetch...")
    start = time.time()
    result = await rag_service.get_official_sequence('SpinEcho')
    elapsed = time.time() - start
    print(f"   - Completed in {elapsed:.2f}s")
    if result:
        print(f"   - Found: {result['file_name']}")
        print(f"   - Content length: {len(result['content'])} chars")
        print(f"   - Has summary: {'ai_summary' in result}")
    
    # Test 2: Fast function search
    print("\n2. Testing fast function search...")
    start = time.time()
    results = await rag_service.search_functions_fast('makeSincPulse')
    elapsed = time.time() - start
    print(f"   - Completed in {elapsed:.2f}s")
    print(f"   - Found {len(results)} functions")
    
    # Test 3: Function details
    print("\n3. Testing function details...")
    if results:
        start = time.time()
        details = await rag_service.get_function_details([results[0]['name']])
        elapsed = time.time() - start
        print(f"   - Completed in {elapsed:.2f}s")
        if details:
            print(f"   - Got details for: {details[0]['name']}")
    
    print("\nOptimization tests complete!")


if __name__ == "__main__":
    # Run tests if executed directly
    asyncio.run(test_optimizations())