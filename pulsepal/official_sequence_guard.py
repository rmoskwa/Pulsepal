"""
Guard against RECITATION when displaying official Pulseq repository sequences.

This module prevents full code display of official sequences to avoid copyright
and RECITATION issues while still providing helpful information to users.
"""

import logging
from typing import Dict, Optional, Tuple
import re

logger = logging.getLogger(__name__)


class OfficialSequenceGuard:
    """
    Protects against RECITATION when handling official Pulseq sequences.
    """
    
    # Official repository patterns
    OFFICIAL_REPO_PATTERNS = [
        r'github\.com/pulseq/pulseq',
        r'pulseq/pulseq',
        r'official_sequence_examples',
    ]
    
    # Keywords that indicate user wants full/combined code
    FULL_CODE_INDICATORS = [
        'combine',
        'full code',
        'complete code',
        'single block',
        'one block',
        'entire code',
        'whole sequence',
        'all together',
        'without chunks',
        'uncombined'
    ]
    
    @classmethod
    def is_official_sequence(cls, source: str = None, url: str = None, 
                            table_name: str = None) -> bool:
        """
        Check if content is from official Pulseq repository.
        
        Args:
            source: Source identifier
            url: URL of the content
            table_name: Database table name
            
        Returns:
            True if content is from official repository
        """
        # Check if from official_sequence_examples view
        if table_name == "official_sequence_examples":
            return True
            
        # Check URL patterns
        if url:
            for pattern in cls.OFFICIAL_REPO_PATTERNS:
                if re.search(pattern, url, re.IGNORECASE):
                    return True
                    
        # Check source patterns
        if source:
            for pattern in cls.OFFICIAL_REPO_PATTERNS:
                if re.search(pattern, source, re.IGNORECASE):
                    return True
                    
        return False
    
    @classmethod
    def user_wants_full_code(cls, query: str) -> bool:
        """
        Detect if user is asking for complete/combined code.
        
        Args:
            query: User's query
            
        Returns:
            True if user wants full code display
        """
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in cls.FULL_CODE_INDICATORS)
    
    @classmethod
    def format_protected_response(cls, sequence_info: Dict, 
                                 include_preview: bool = True) -> str:
        """
        Format response for protected official sequences.
        
        Args:
            sequence_info: Information about the sequence
            include_preview: Whether to include a code preview
            
        Returns:
            Formatted response explaining the limitation
        """
        sequence_type = sequence_info.get('sequence_type', 'sequence')
        file_name = sequence_info.get('file_name', '')
        url = sequence_info.get('url', '')
        ai_summary = sequence_info.get('ai_summary', '')
        
        response = f"""I understand you'd like the complete code in one block. Since this is an official Pulseq repository sequence ({sequence_type}), I've presented it in educational sections above to avoid potential copyright concerns and ensure proper understanding.

**You can access the complete original code at:**
{url if url else f'`{file_name}` in the Pulseq repository'}

The sectioned presentation above contains all the key components. If you need a working version for your specific application, I can help you:
- Create a custom implementation based on your parameters
- Modify specific sections for your scanner configuration  
- Explain any part in more detail
- Combine YOUR custom modifications into a single file

Would you like me to help adapt this sequence for your specific needs?"""
        
        if ai_summary:
            response += f"\n\n### Sequence Overview\n{ai_summary}"
            
        if include_preview:
            response += """

### Getting Started
To use this sequence:
1. Clone the Pulseq repository: `git clone https://github.com/pulseq/pulseq.git`
2. Navigate to the sequence file
3. Modify parameters for your application

Would you like me to:
- Explain specific parts of the sequence?
- Help you modify it for your scanner/application?
- Show you how to implement a similar custom sequence?"""
        
        return response
    
    @classmethod
    def check_and_guard(cls, query: str, source_info: Dict) -> Tuple[bool, Optional[str]]:
        """
        Check if response needs RECITATION protection and return appropriate message.
        
        Args:
            query: User's query
            source_info: Information about the content source
            
        Returns:
            Tuple of (needs_protection, protection_message)
        """
        # Check if this is official content
        is_official = cls.is_official_sequence(
            source=source_info.get('source'),
            url=source_info.get('url'),
            table_name=source_info.get('table_name')
        )
        
        if not is_official:
            return False, None
            
        # Check if user wants full code
        wants_full = cls.user_wants_full_code(query)
        
        if wants_full:
            logger.info(f"Protecting official sequence from full display: {source_info.get('sequence_type')}")
            message = cls.format_protected_response(source_info)
            return True, message
            
        return False, None
    
    @classmethod
    def add_attribution(cls, code_block: str, source_info: Dict) -> str:
        """
        Add proper attribution to code snippets.
        
        Args:
            code_block: The code to attribute
            source_info: Information about the source
            
        Returns:
            Code block with attribution
        """
        attribution = f"% Source: {source_info.get('url', 'Pulseq Official Repository')}\n"
        attribution += f"% File: {source_info.get('file_name', 'Unknown')}\n"
        attribution += "% Copyright: Pulseq Project Contributors\n"
        attribution += "% License: See Pulseq repository for license information\n\n"
        
        return attribution + code_block


# Global instance
_guard = None

def get_official_sequence_guard() -> OfficialSequenceGuard:
    """Get or create the global official sequence guard."""
    global _guard
    if _guard is None:
        _guard = OfficialSequenceGuard()
    return _guard