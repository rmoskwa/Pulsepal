"""Environment variable validation for PulsePal deployment."""

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple


class EnvValidator:
    """Validates environment variables for PulsePal deployment."""

    # Required environment variables for PulsePal
    REQUIRED_VARS = {
        "GOOGLE_API_KEY": {
            "description": "Google AI/Gemini API key",
            "example": "AIza...",
            "validation": lambda v: v and v.startswith("AIza") and len(v) > 20,
        },
        "SUPABASE_URL": {
            "description": "Supabase project URL",
            "example": "https://xxx.supabase.co",
            "validation": lambda v: v
            and v.startswith("https://")
            and "supabase.co" in v,
        },
        "SUPABASE_KEY": {
            "description": "Supabase anon/public key",
            "example": "eyJ...",
            "validation": lambda v: v and len(v) > 50,
        },
        "CHAINLIT_AUTH_SECRET": {
            "description": "Secret for Chainlit authentication",
            "example": "your-secret-key-here",
            "validation": lambda v: v and len(v) >= 16,
        },
    }

    # Optional but recommended variables
    OPTIONAL_VARS = {
        "GOOGLE_EMBEDDINGS_API_KEY": {
            "description": "Google Embeddings API key (defaults to GOOGLE_API_KEY)",
            "example": "AIza...",
            "validation": lambda v: not v or (v.startswith("AIza") and len(v) > 20),
        },
        "RAILWAY_PUBLIC_DOMAIN": {
            "description": "Railway public domain for deployment",
            "example": "web-production-xxxx.up.railway.app",
            "validation": lambda v: not v or ("railway.app" in v),
        },
        "ADMIN_API_KEY": {
            "description": "Admin API key for protected endpoints",
            "example": "admin-key-xxxx",
            "validation": lambda v: not v or len(v) >= 10,
        },
        "ALPHA_API_KEYS": {
            "description": "JSON string of alpha tester API keys",
            "example": '{"key1": {"name": "User", "email": "user@example.com"}}',
            "validation": lambda v: not v or EnvValidator._validate_json(v),
        },
        "OAUTH_GOOGLE_CLIENT_ID": {
            "description": "Google OAuth client ID",
            "example": "xxxx.apps.googleusercontent.com",
            "validation": lambda v: not v or "apps.googleusercontent.com" in v,
        },
        "OAUTH_GOOGLE_CLIENT_SECRET": {
            "description": "Google OAuth client secret",
            "example": "GOCSPX-...",
            "validation": lambda v: not v or v.startswith("GOCSPX-"),
        },
    }

    @staticmethod
    def _validate_json(value: str) -> bool:
        """Validate JSON string."""
        try:
            json.loads(value)
            return True
        except (json.JSONDecodeError, TypeError, ValueError):
            return False

    @staticmethod
    def _mask_value(value: str, show_chars: int = 4) -> str:
        """Mask sensitive values for display."""
        if not value or len(value) <= show_chars * 2:
            return "***"
        return f"{value[:show_chars]}...{value[-show_chars:]}"

    @classmethod
    def validate_required(cls, verbose: bool = False) -> Tuple[bool, List[str]]:
        """Validate all required environment variables."""
        missing = []
        invalid = []

        for var_name, config in cls.REQUIRED_VARS.items():
            value = os.environ.get(var_name)

            if not value:
                missing.append(var_name)
                if verbose:
                    print(f"❌ {var_name}: MISSING")
                    print(f"   Description: {config['description']}")
                    print(f"   Example: {config['example']}")
            elif not config["validation"](value):
                invalid.append(var_name)
                if verbose:
                    print(f"⚠️  {var_name}: INVALID FORMAT")
                    print(f"   Description: {config['description']}")
                    print(f"   Example: {config['example']}")
                    print(f"   Current: {cls._mask_value(value)}")
            elif verbose:
                print(f"✅ {var_name}: OK ({cls._mask_value(value)})")

        is_valid = len(missing) == 0 and len(invalid) == 0
        return is_valid, missing + invalid

    @classmethod
    def validate_optional(cls, verbose: bool = False) -> Dict[str, bool]:
        """Validate optional environment variables."""
        results = {}

        if verbose:
            print("\nOptional Variables:")

        for var_name, config in cls.OPTIONAL_VARS.items():
            value = os.environ.get(var_name)

            if not value:
                results[var_name] = None
                if verbose:
                    print(f"⚪ {var_name}: NOT SET (optional)")
            elif config["validation"](value):
                results[var_name] = True
                if verbose:
                    print(f"✅ {var_name}: OK ({cls._mask_value(value)})")
            else:
                results[var_name] = False
                if verbose:
                    print(f"⚠️  {var_name}: INVALID FORMAT")
                    print(f"   Description: {config['description']}")
                    print(f"   Example: {config['example']}")

        return results

    @classmethod
    def validate_all(cls, verbose: bool = False) -> bool:
        """Validate all environment variables."""
        if verbose:
            print("=" * 60)
            print("PulsePal Environment Validation")
            print("=" * 60)
            print(f"Timestamp: {datetime.now().isoformat()}")
            print(f"Environment: {os.environ.get('RAILWAY_ENVIRONMENT', 'local')}")
            print("\nRequired Variables:")

        # Validate required
        is_valid, issues = cls.validate_required(verbose)

        # Validate optional
        cls.validate_optional(verbose)

        if verbose:
            print("\n" + "=" * 60)
            if is_valid:
                print("✅ All required environment variables are valid!")
            else:
                print("❌ Environment validation FAILED!")
                print(f"   Issues with: {', '.join(issues)}")
            print("=" * 60)

        return is_valid

    @classmethod
    def get_status_dict(cls, include_values: bool = False) -> Dict:
        """Get validation status as dictionary."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "environment": os.environ.get("RAILWAY_ENVIRONMENT", "local"),
            "is_valid": True,
            "required": {},
            "optional": {},
        }

        # Check required variables
        for var_name, config in cls.REQUIRED_VARS.items():
            value = os.environ.get(var_name)
            var_status = {
                "present": bool(value),
                "valid": bool(value and config["validation"](value)),
                "description": config["description"],
            }

            if include_values and value:
                var_status["value"] = cls._mask_value(value, show_chars=8)

            status["required"][var_name] = var_status
            if not var_status["valid"]:
                status["is_valid"] = False

        # Check optional variables
        for var_name, config in cls.OPTIONAL_VARS.items():
            value = os.environ.get(var_name)
            var_status = {
                "present": bool(value),
                "valid": not value or config["validation"](value),
                "description": config["description"],
            }

            if include_values and value:
                var_status["value"] = cls._mask_value(value, show_chars=8)

            status["optional"][var_name] = var_status

        return status

    @classmethod
    def get_missing_vars(cls) -> List[str]:
        """Get list of missing required variables."""
        missing = []
        for var_name in cls.REQUIRED_VARS:
            if not os.environ.get(var_name):
                missing.append(var_name)
        return missing

    @classmethod
    def generate_env_template(cls, include_current: bool = False) -> str:
        """Generate .env template with all variables."""
        lines = [
            "# PulsePal Environment Variables",
            "# Generated on: " + datetime.now().isoformat(),
            "",
            "# REQUIRED VARIABLES",
            "# ==================",
            "",
        ]

        for var_name, config in cls.REQUIRED_VARS.items():
            lines.append(f"# {config['description']}")
            lines.append(f"# Example: {config['example']}")
            if include_current and os.environ.get(var_name):
                lines.append(f"{var_name}={cls._mask_value(os.environ.get(var_name))}")
            else:
                lines.append(f"{var_name}=")
            lines.append("")

        lines.extend(["", "# OPTIONAL VARIABLES", "# ==================", ""])

        for var_name, config in cls.OPTIONAL_VARS.items():
            lines.append(f"# {config['description']}")
            lines.append(f"# Example: {config['example']}")
            if include_current and os.environ.get(var_name):
                lines.append(f"{var_name}={cls._mask_value(os.environ.get(var_name))}")
            else:
                lines.append(f"# {var_name}=")
            lines.append("")

        return "\n".join(lines)


# Quick validation function for scripts
def validate_deployment_environment(silent: bool = False) -> bool:
    """Quick validation function for deployment scripts."""
    if not silent:
        print("\nValidating deployment environment...")

    is_valid = EnvValidator.validate_all(verbose=not silent)

    if not is_valid and not silent:
        missing = EnvValidator.get_missing_vars()
        print(f"\n⚠️  Missing required environment variables: {', '.join(missing)}")
        print("Please set these in your .env file or Railway dashboard.")

    return is_valid
