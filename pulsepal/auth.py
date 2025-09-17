# pulsepal/auth.py
"""
Database-backed API key authentication with message limits.
"""

import os
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
from supabase import create_client, Client

# Set up logging
logger = logging.getLogger(__name__)


class APIKeyAuth:
    """Database-backed API key authentication."""

    def __init__(self):
        """Initialize Supabase client for API key management."""
        # Import settings here to avoid circular imports
        from .settings import get_settings

        settings = get_settings()
        self.supabase: Client = create_client(
            settings.supabase_url, settings.supabase_key
        )
        # Cache for performance
        self._key_cache: Dict[str, Tuple[dict, datetime]] = {}
        self._cache_duration = 300  # 5 minutes

    def validate_api_key(self, api_key: str) -> Optional[dict]:
        """Validate API key and return user info if valid."""
        # Check cache first
        if api_key in self._key_cache:
            cached_data, cached_time = self._key_cache[api_key]
            if (datetime.now() - cached_time).seconds < self._cache_duration:
                return cached_data

        # Query database
        try:
            response = (
                self.supabase.table("api_keys")
                .select("*")
                .eq("api_key", api_key)
                .eq("is_active", True)
                .execute()
            )

            if not response.data:
                return None

            key_data = response.data[0]

            # Check if expired
            if key_data.get("expires_at"):
                expires_at = datetime.fromisoformat(
                    key_data["expires_at"].replace("Z", "+00:00")
                )
                if expires_at < datetime.now():
                    return None

            # Check message limit
            if key_data["messages_used"] >= key_data["message_limit"]:
                logger.warning(f"API key {api_key[:10]}... has reached message limit")
                return None

            # Format response
            user_info = {
                "id": key_data["id"],
                "name": key_data["username"],
                "email": key_data.get("email", f"{key_data['username']}@pulsepal.ai"),
                "limit": key_data["message_limit"],
                "used": key_data["messages_used"],
                "remaining": key_data["message_limit"] - key_data["messages_used"],
                "organization": key_data.get("organization"),
                "created": key_data["created_at"],
            }

            # Cache the result
            self._key_cache[api_key] = (user_info, datetime.now())

            return user_info

        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            # Fall back to environment variable keys if database is unavailable
            return self._fallback_validation(api_key)

    def _fallback_validation(self, api_key: str) -> Optional[dict]:
        """Fallback to environment variable API keys if database is unavailable."""
        api_keys_str = os.getenv("ALPHA_API_KEYS")

        if not api_keys_str:
            # Default test key for local development
            if api_key == "test-key":
                return {
                    "name": "Test User",
                    "email": "test@example.com",
                    "limit": 1000,
                    "used": 0,
                    "remaining": 1000,
                    "created": "2025-01-01",
                }
            return None

        # Remove quotes if present
        api_keys_str = api_keys_str.strip('"').strip("'")

        # Check if key is in environment variable list
        env_keys = api_keys_str.split(",")
        if api_key in [k.strip() for k in env_keys]:
            return {
                "name": "Alpha User",
                "email": "user@alpha.pulsepal.ai",
                "limit": 100,
                "used": 0,
                "remaining": 100,
                "created": "2025-01-01",
            }

        return None

    async def increment_usage(self, api_key: str) -> bool:
        """Increment usage count for an API key."""
        try:
            # Get the current API key data
            key_response = (
                self.supabase.table("api_keys")
                .select("*")
                .eq("api_key", api_key)
                .execute()
            )

            if not key_response.data:
                return False

            key_data = key_response.data[0]

            # Check if already at limit
            if key_data["messages_used"] >= key_data["message_limit"]:
                logger.warning(f"API key {api_key[:10]}... already at message limit")
                return False

            # Update usage count and last used timestamp
            update_response = (
                self.supabase.table("api_keys")
                .update(
                    {
                        "messages_used": key_data["messages_used"] + 1,
                        "last_used_at": datetime.now().isoformat(),
                    }
                )
                .eq("api_key", api_key)
                .execute()
            )

            # Clear cache for this key
            if api_key in self._key_cache:
                del self._key_cache[api_key]

            return bool(update_response.data)

        except Exception as e:
            logger.error(f"Error incrementing usage: {e}")
            return False

    def check_rate_limit(self, api_key: str, limit: int = 100) -> bool:
        """Simple rate limit check based on last_used_at timestamp."""
        # For now, always return True since we don't have detailed logs
        # In production, you might want to use Redis or in-memory cache for rate limiting
        return True


# Global instance
_auth_instance = None


def get_auth() -> APIKeyAuth:
    """Get or create the global APIKeyAuth instance."""
    global _auth_instance
    if _auth_instance is None:
        _auth_instance = APIKeyAuth()
    return _auth_instance


# Compatibility functions for existing code
def validate_api_key(api_key: str) -> Optional[dict]:
    """Validate API key and return user info if valid."""
    return get_auth().validate_api_key(api_key)


def check_rate_limit(api_key: str, limit: int = 100) -> bool:
    """Simple rate limiting check (requests per hour)."""
    return get_auth().check_rate_limit(api_key, limit)


# Chainlit-specific auth callback - only import when needed
def get_chainlit_auth_callback():
    """Get Chainlit auth callback - delays import until needed."""
    import chainlit as cl

    @cl.password_auth_callback
    def auth_callback(username: str, password: str) -> Optional[cl.User]:
        """Authenticate user with API key as password."""
        user_info = validate_api_key(password)

        if user_info:
            return cl.User(
                identifier=username or user_info["email"],
                metadata={
                    "api_key": password,
                    "api_key_id": user_info.get("id"),
                    "name": user_info["name"],
                    "limit": user_info["limit"],
                    "used": user_info.get("used", 0),
                    "remaining": user_info.get("remaining", user_info["limit"]),
                    "organization": user_info.get("organization"),
                },
            )
        return None

    return auth_callback
