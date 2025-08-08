#!/usr/bin/env python3
"""
Apply optimizations to PulsePal RAG service at startup.

This module patches the existing RAG service with optimized methods
to improve performance when fetching large content fields.
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def apply_rag_optimizations():
    """
    Apply optimizations to the RAG service when it's initialized.
    This should be called after the RAG service is created.
    """
    from .rag_service import get_rag_service
    
    rag_service = get_rag_service()
    
    # Check if optimizations already applied
    if hasattr(rag_service, '_optimizations_applied'):
        logger.debug("RAG optimizations already applied")
        return rag_service
    
    logger.info("Applying RAG optimizations...")
    
    # DISABLED: The semantic search implementation in rag_service.py is now the optimized version
    # We don't want to override it with the old exact-match approach
    
    # Mark as optimized to prevent re-application
    rag_service._optimizations_applied = True
    logger.info("RAG optimizations applied successfully")
    return rag_service
    
    # ===== DISABLED CODE BELOW =====
    
    # Store original methods
    rag_service._original_get_official_sequence = rag_service.get_official_sequence
    
    # Apply optimized get_official_sequence
    async def get_official_sequence_optimized(sequence_type: str) -> Optional[Dict[str, Any]]:
        """Optimized version with proper two-tier fetching."""
        try:
            # Use the existing normalization
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
            
            # Two-step query to avoid fetching large content during search
            # Step 1: Get file_name and metadata (file_name is unique identifier)
            summary_result = rag_service.supabase_client.client.from_('official_sequence_examples')\
                .select('file_name, sequence_type, ai_summary')\
                .eq('sequence_type', normalized_type)\
                .limit(1)\
                .execute()
            
            if not summary_result.data:
                return None
            
            record = summary_result.data[0]
            file_name = record['file_name']
            
            # Step 2: Fetch content for specific record using file_name
            content_result = rag_service.supabase_client.client.from_('official_sequence_examples')\
                .select('content')\
                .eq('file_name', file_name)\
                .limit(1)\
                .execute()
            
            if content_result.data and content_result.data[0].get('content'):
                return {
                    'content': content_result.data[0]['content'],
                    'file_name': record['file_name'],
                    'sequence_type': record['sequence_type'],
                    'ai_summary': record.get('ai_summary', '')
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Optimized get_official_sequence failed: {e}")
            # Fallback to original
            return await rag_service._original_get_official_sequence(sequence_type)
    
    # Replace method
    rag_service.get_official_sequence = get_official_sequence_optimized
    
    # Mark as optimized
    rag_service._optimizations_applied = True
    
    logger.info("RAG optimizations applied successfully")
    return rag_service