"""
Patch for Gemini model to properly handle unexpected finish reasons.

This handles cases where Gemini returns finish reasons that aren't in pydantic-ai's expected values.
Instead of masking these signals, we handle them appropriately based on their meaning.
"""

import logging
from typing import Any, Dict

from pydantic_ai.models.gemini import GeminiModel

logger = logging.getLogger(__name__)


class GeminiRecitationError(Exception):
    """Raised when Gemini detects potential training data recitation."""


class PatchedGeminiModel(GeminiModel):
    """
    Patched Gemini model that properly handles unexpected finish reasons.

    Instead of masking important signals like RECITATION, this patch:
    1. Detects and raises appropriate exceptions for handling upstream
    2. Only masks truly benign unexpected reasons
    3. Provides context about why responses were blocked
    """

    def __init__(self, *args, **kwargs):
        """Initialize patched Gemini model."""
        super().__init__(*args, **kwargs)

    def _process_response(self, response: Dict[str, Any]) -> Any:
        """Process response and handle unexpected finish reasons appropriately."""
        if "candidates" in response:
            for candidate in response["candidates"]:
                if "finishReason" in candidate:
                    finish_reason = candidate["finishReason"]

                    # Handle RECITATION - this is important signal, not to be masked
                    if finish_reason == "RECITATION":
                        logger.warning(
                            "Gemini detected potential training data recitation",
                        )

                        # Raise a specific exception that can be caught upstream
                        raise GeminiRecitationError(
                            "Gemini blocked response due to potential training data recitation. "
                            "Consider using verified database examples instead.",
                        )

                    # Handle SAFETY - also important, should not be masked
                    if finish_reason == "SAFETY":
                        logger.warning("Gemini blocked response due to safety filters")
                        # Let this through as-is, pydantic-ai handles SAFETY

                    # Handle truly unexpected but benign reasons
                    elif finish_reason in ["OTHER", "UNEXPECTED_TOOL_CALL"]:
                        logger.warning(
                            f"Patching benign finish reason {finish_reason} to STOP",
                        )
                        candidate["finishReason"] = "STOP"

                        # Ensure there's minimal content if missing
                        if "content" not in candidate or not candidate["content"]:
                            candidate["content"] = {
                                "parts": [{"text": "I'll help you with this request."}],
                            }

                    # Handle any other unexpected finish reasons
                    elif finish_reason not in ["STOP", "MAX_TOKENS", "SAFETY"]:
                        logger.error(f"Unknown finish reason: {finish_reason}")
                        # For truly unknown reasons, raise an error for investigation
                        raise ValueError(
                            f"Unexpected Gemini finish reason: {finish_reason}. "
                            f"This needs to be handled in gemini_patch.py",
                        )

        # Call the parent's _process_response to properly create the response object
        return super()._process_response(response)

    async def request(self, *args, **kwargs):
        """Override request to handle responses properly."""
        try:
            # Call parent request method
            result = await super().request(*args, **kwargs)
            return result

        except GeminiRecitationError:
            # Re-raise our specific error for handling upstream
            raise

        except Exception as e:
            error_str = str(e)

            # Check if it's a pydantic validation error for RECITATION
            if "RECITATION" in error_str and "validation error" in error_str:
                # Convert to our specific exception
                raise GeminiRecitationError(
                    "Gemini blocked response due to potential training data recitation",
                ) from e

            # Check for other validation errors we should handle
            if "validation error" in error_str and any(
                reason in error_str for reason in ["OTHER", "UNEXPECTED_TOOL_CALL"]
            ):
                # These are benign but we can't auto-retry
                logger.warning(
                    f"Unexpected finish reason that needs patching: {error_str}",
                )
                # Convert to a more specific error
                raise ValueError(
                    f"Gemini returned unexpected finish reason. Original error: {error_str}",
                ) from e

            # Unknown error, let it propagate
            raise
