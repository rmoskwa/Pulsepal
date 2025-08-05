"""
Patch for Gemini model to handle UNEXPECTED_TOOL_CALL finish reason.
"""

import logging
from typing import Any, Dict
from pydantic_ai.models.gemini import GeminiModel

logger = logging.getLogger(__name__)


class PatchedGeminiModel(GeminiModel):
    """
    Patched Gemini model that handles UNEXPECTED_TOOL_CALL finish reason.
    
    This is a workaround for the issue where Gemini returns 'UNEXPECTED_TOOL_CALL'
    as a finish reason, which isn't in pydantic-ai's expected values.
    """
    
    def _process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process and patch the response if needed."""
        # Check if response has candidates
        if 'candidates' in response:
            for candidate in response['candidates']:
                if 'finishReason' in candidate:
                    finish_reason = candidate['finishReason']
                    
                    # If we see UNEXPECTED_TOOL_CALL, change it to STOP
                    if finish_reason == 'UNEXPECTED_TOOL_CALL':
                        logger.warning(
                            "Patching UNEXPECTED_TOOL_CALL finish reason to STOP"
                        )
                        candidate['finishReason'] = 'STOP'
                        
                        # Also ensure there's content if missing
                        if 'content' not in candidate or not candidate['content']:
                            candidate['content'] = {
                                'parts': [{
                                    'text': "I understand your request, but I'm unable to use tools in this context. Let me help you directly instead."
                                }]
                            }
        
        return response
    
    async def request(self, *args, **kwargs):
        """Override request to patch responses."""
        try:
            # Call parent request method
            result = await super().request(*args, **kwargs)
            return result
        except Exception as e:
            # If it's the specific validation error, try to handle it
            if "UNEXPECTED_TOOL_CALL" in str(e):
                logger.error(f"Caught UNEXPECTED_TOOL_CALL error: {e}")
                # Re-raise for now - would need deeper integration to fix
            raise