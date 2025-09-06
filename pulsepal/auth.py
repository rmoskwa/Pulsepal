# pulsepal/auth.py
"""
Simple API key authentication for alpha testing.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional

# Set up logging
logger = logging.getLogger(__name__)


# Load API keys from environment or file
def load_api_keys() -> Dict[str, dict]:
    """Load API keys from environment variable or file."""
    logger.info("Loading API keys...")
    api_keys_str = os.getenv("ALPHA_API_KEYS")

    if api_keys_str:
        logger.info(
            f"Found ALPHA_API_KEYS environment variable (length: {len(api_keys_str)})"
        )

        # Remove quotes if present
        api_keys_str = api_keys_str.strip('"').strip("'")

        # First try to parse as JSON
        try:
            keys = json.loads(api_keys_str)
            logger.info(f"Successfully parsed {len(keys)} API keys from JSON format")
            return keys
        except json.JSONDecodeError:
            # If not JSON, treat as comma-separated list
            logger.info("Parsing as comma-separated list of API keys")
            keys = {}
            for i, key in enumerate(api_keys_str.split(",")):
                key = key.strip()
                if key:
                    keys[key] = {
                        "name": f"Alpha User {i+1}",
                        "email": f"user{i+1}@alpha.pulsepal.ai",
                        "limit": 100,
                        "created": "2025-01-01",
                    }
            logger.info(f"Parsed {len(keys)} API keys from comma-separated format")
            return (
                keys
                if keys
                else {
                    "test-key": {
                        "name": "Test User",
                        "email": "test@example.com",
                        "limit": 1000,
                        "created": "2025-01-01",
                    }
                }
            )
    else:
        logger.info(
            "No ALPHA_API_KEYS environment variable found, using default test key"
        )

    # Default for local testing
    return {
        "test-key": {
            "name": "Test User",
            "email": "test@example.com",
            "limit": 1000,
            "created": "2025-01-01",
        }
    }


API_KEYS = load_api_keys()

# Rate limiting tracker (in-memory for alpha)
RATE_LIMITS: Dict[str, list] = {}


def validate_api_key(api_key: str) -> Optional[dict]:
    """Validate API key and return user info if valid."""
    return API_KEYS.get(api_key)


def check_rate_limit(api_key: str, limit: int = 100) -> bool:
    """Simple rate limiting check (requests per hour)."""
    now = datetime.now()
    hour_ago = now.timestamp() - 3600

    if api_key not in RATE_LIMITS:
        RATE_LIMITS[api_key] = []

    # Clean old entries
    RATE_LIMITS[api_key] = [ts for ts in RATE_LIMITS[api_key] if ts > hour_ago]

    # Check limit
    if len(RATE_LIMITS[api_key]) >= limit:
        return False

    # Add current request
    RATE_LIMITS[api_key].append(now.timestamp())
    return True


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
                    "name": user_info["name"],
                    "limit": user_info["limit"],
                },
            )
        return None

    return auth_callback
