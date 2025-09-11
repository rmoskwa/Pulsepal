"""
SQL query validation layer for PulsePal.
Provides intelligent error handling and guidance for Gemini's Supabase queries.
"""

import logging
import re
import threading
from typing import Any, Dict, List


logger = logging.getLogger(__name__)

# Valid tables in the PulsePal knowledge base
VALID_TABLES = [
    "api_reference",
    "pulseq_sequences",
    "sequence_chunks",
    "crawled_code",
    "crawled_docs",
    "table_metadata",
]

# Table schemas cache (populated on first use)
_table_schemas_cache: Dict[str, List[str]] = {}
_cache_lock = threading.Lock()  # Thread-safe access to cache

# Common PostgreSQL error codes
ERROR_CODES = {
    "42P01": "table_not_found",
    "42703": "column_not_found",
    "22P02": "type_mismatch",
    "42601": "syntax_error",
    "42883": "function_not_found",
}


class SupabaseQueryValidator:
    """Validates and provides guidance for Supabase client queries."""

    @staticmethod
    def parse_postgresql_error(error_message: str) -> Dict[str, Any]:
        """
        Parse PostgreSQL error messages from Supabase.

        Args:
            error_message: Raw error message from Supabase

        Returns:
            Parsed error information with code, message, hints
        """
        # Handle None or empty error messages
        if not error_message:
            return {
                "raw_error": "No error message provided",
                "error_code": None,
                "error_type": "unknown",
                "details": {},
            }

        # Convert to string if not already
        error_message = str(error_message)

        parsed = {
            "raw_error": error_message,
            "error_code": None,
            "error_type": None,
            "details": {},
        }

        # Check if this is a dict-like error from Supabase Python client
        if "{'message':" in error_message or '{"message":' in error_message:
            try:
                import ast
                import json

                # Try to parse as Python dict or JSON
                if error_message.startswith("{"):
                    try:
                        error_dict = ast.literal_eval(error_message)
                    except (ValueError, SyntaxError):
                        error_dict = json.loads(error_message)

                    if isinstance(error_dict, dict):
                        # Extract from Supabase error format
                        if "code" in error_dict:
                            parsed["error_code"] = error_dict["code"]
                            parsed["error_type"] = ERROR_CODES.get(
                                error_dict["code"], "unknown"
                            )
                        if "message" in error_dict:
                            error_message = error_dict[
                                "message"
                            ]  # Update to use the actual message
                        if "hint" in error_dict and error_dict["hint"]:
                            parsed["details"]["hint"] = error_dict["hint"]
            except (ValueError, SyntaxError, json.JSONDecodeError, ImportError):
                pass  # Fall back to string parsing

        # Extract PostgreSQL error code from string format
        if not parsed["error_code"]:
            code_match = re.search(r"ERROR:\s*(\w+):", error_message)
            if code_match:
                error_code = code_match.group(1)
                parsed["error_code"] = error_code
                parsed["error_type"] = ERROR_CODES.get(error_code, "unknown")

        # Extract specific error patterns
        if "relation" in error_message and "does not exist" in error_message:
            table_match = re.search(r'relation "(.+?)" does not exist', error_message)
            if table_match:
                parsed["details"]["missing_table"] = table_match.group(1)
                parsed["error_type"] = "table_not_found"

        elif "column" in error_message and "does not exist" in error_message:
            # Try different patterns for column extraction
            column_match = re.search(
                r'column "?([^"]+)"? does not exist', error_message
            )
            if not column_match:
                column_match = re.search(
                    r"column ([a-zA-Z_][a-zA-Z0-9_.]*) does not exist", error_message
                )

            if column_match:
                full_column = column_match.group(1)
                # Extract just the column name (after the table name if present)
                if "." in full_column:
                    parsed["details"]["missing_column"] = full_column.split(".")[-1]
                else:
                    parsed["details"]["missing_column"] = full_column
                parsed["error_type"] = "column_not_found"

            # Check for PostgreSQL's helpful hints
            hint_match = re.search(
                r'HINT:\s*Perhaps you meant to reference the column "(.+?)"',
                error_message,
            )
            if hint_match:
                parsed["details"]["suggested_column"] = hint_match.group(1)
            elif "hint" in parsed["details"]:
                # Check if hint contains a suggestion
                hint = parsed["details"]["hint"]
                if hint and "Perhaps you meant" in hint:
                    suggestion_match = re.search(r'"([^"]+)"', hint)
                    if suggestion_match:
                        parsed["details"]["suggested_column"] = suggestion_match.group(
                            1
                        )

        elif "invalid input syntax" in error_message:
            type_match = re.search(
                r'invalid input syntax for type (\w+):\s*"(.+?)"', error_message
            )
            if type_match:
                parsed["details"]["expected_type"] = type_match.group(1)
                parsed["details"]["invalid_value"] = type_match.group(2)
                parsed["error_type"] = "type_mismatch"

        elif "malformed array literal" in error_message:
            # Handle malformed array literal errors
            value_match = re.search(
                r'malformed array literal:\s*"(.+?)"', error_message
            )
            if value_match:
                parsed["details"]["invalid_array_value"] = value_match.group(1)
                parsed["error_type"] = "malformed_array"
                parsed["details"]["hint"] = (
                    "Array values must be provided as lists, not strings"
                )

        elif "function" in error_message and "does not exist" in error_message:
            func_match = re.search(r"function (.+?) does not exist", error_message)
            if func_match:
                parsed["details"]["missing_function"] = func_match.group(1)
                parsed["error_type"] = "function_not_found"

        elif "operator does not exist" in error_message:
            # Handle array operator errors
            operator_match = re.search(r"operator does not exist: (.+)", error_message)
            if operator_match:
                parsed["details"]["invalid_operator"] = operator_match.group(1)
                parsed["error_type"] = "invalid_operator"
                # Check if it's an array field error
                if "[]" in operator_match.group(
                    1
                ) or "bigint[]" in operator_match.group(1):
                    parsed["error_type"] = "array_operator_error"
                    parsed["details"]["is_array_field"] = True

        return parsed

    @staticmethod
    def generate_error_guidance(
        parsed_error: Dict[str, Any], context: Dict[str, Any] = None
    ) -> str:
        """
        Generate helpful guidance based on parsed error.

        Args:
            parsed_error: Parsed error information
            context: Additional context (table name, available columns, etc.)

        Returns:
            Guidance message for ModelRetry
        """
        error_type = parsed_error.get("error_type")
        details = parsed_error.get("details", {})

        if error_type == "table_not_found":
            table = details.get("missing_table", "unknown")
            return (
                f"❌ **Table '{table}' not found**\n\n"
                f"Valid tables in the knowledge base:\n"
                f"{format_table_list(VALID_TABLES)}\n\n"
                f"**Try one of these approaches:**\n"
                f"1. Use `find_relevant_tables(query='your search')` to discover appropriate tables\n"
                f"2. Use `get_table_schemas(tables=['table_name'])` to explore table structure\n"
                f"3. Check the table name spelling - PostgreSQL is case-sensitive"
            )

        elif error_type == "column_not_found":
            column = details.get("missing_column", "unknown")
            suggestion = details.get("suggested_column")
            table = context.get("table") if context else None

            message = f"❌ **Column '{column}' not found**"

            if suggestion:
                message += f"\n\n✅ **Did you mean:** `{suggestion}`?"

            if table and context and "valid_columns" in context:
                valid_cols = context["valid_columns"]
                message += f"\n\n**Valid columns in '{table}':**\n"
                message += format_column_list(valid_cols[:15])  # Show first 15
                if len(valid_cols) > 15:
                    message += f"\n... and {len(valid_cols) - 15} more columns"
                message += f"\n\n**Tip:** Use `get_table_schemas(['{table}'])` to see all columns with descriptions"
            else:
                message += "\n\n**To see valid columns:**\n"
                message += "Use `get_table_schemas(['table_name'])` to explore the table structure"

            return message

        elif error_type == "type_mismatch":
            expected = details.get("expected_type", "unknown")
            invalid = details.get("invalid_value", "unknown")

            type_guidance = get_type_guidance(expected, invalid)

            return (
                f"❌ **Type mismatch error**\n\n"
                f"Expected type: **{expected}**\n"
                f"You provided: `'{invalid}'` (string)\n\n"
                f"{type_guidance}"
            )

        elif error_type == "malformed_array":
            invalid_value = details.get("invalid_array_value", "unknown")
            return (
                f"❌ **Malformed array value**\n\n"
                f'Invalid value: `"{invalid_value}"`\n\n'
                f"**For array fields, provide values as Python lists:**\n"
                f'• Single value: `[72]` not `"72"`\n'
                f'• Multiple values: `[72, 73, 74]` not `"72,73,74"`\n\n'
                f"**Correct filter format:**\n"
                "```python\n"
                '{"column": "parent_sequences", "operator": "contains", "value": [72]}\n'
                "```\n\n"
                f"**Note:** The `contains` operator checks if the array field contains the specified value(s)."
            )

        elif error_type == "array_operator_error":
            operator_info = details.get("invalid_operator", "unknown")
            return (
                f"❌ **Invalid operator for array field**\n\n"
                f"Error: {operator_info}\n\n"
                f"**For array fields (like `parent_sequences`), use:**\n"
                f"• `contains` operator with array value:\n"
                "```python\n"
                '{"column": "parent_sequences", "operator": "contains", "value": [72]}\n'
                "```\n\n"
                f"**Common array operations:**\n"
                f"• Check if array contains a value: `contains` with `[value]`\n"
                f"• Check if array contains multiple values: `contains` with `[val1, val2]`\n"
                f"• Check if field is not null: `is` with `not null`\n\n"
                f"**Note:** You cannot use text operators (like, ilike) on array fields.\n"
                f"Array fields require array-specific operators."
            )

        elif error_type == "function_not_found":
            function = details.get("missing_function", "unknown")
            return (
                f"❌ **RPC function '{function}' not found**\n\n"
                f"This RPC function doesn't exist in the database.\n\n"
                f"**Available RPC functions:**\n"
                f"• `get_distinct_column_values` - Get unique values from a column\n\n"
                f"**For general queries:**\n"
                f"Use the query builder pattern instead of RPC:\n"
                "```python\n"
                "query = {\n"
                '    "table": "table_name",\n'
                '    "select": "*",\n'
                '    "filters": [{"column": "value", "operator": "eq", "value": "something"}],\n'
                '    "limit": 10\n'
                "}\n"
                "```"
            )

        else:
            # Generic error with raw message
            return (
                f"❌ **Database query error**\n\n"
                f"Error: {parsed_error.get('raw_error', 'Unknown error')}\n\n"
                f"**Common fixes:**\n"
                f"• Check table and column names (case-sensitive)\n"
                f"• Verify data types match (no quotes for numbers)\n"
                f"• Use `IS NULL` instead of `= NULL` for null checks\n"
                f"• Use `get_table_schemas()` to explore the database structure"
            )


def format_table_list(tables: List[str]) -> str:
    """Format a list of tables for display."""
    return "\n".join(f"• `{table}`" for table in sorted(tables))


def format_column_list(columns: List[str]) -> str:
    """Format a list of columns for display."""
    # Group into 3 columns for compact display
    if len(columns) <= 6:
        return "\n".join(f"• `{col}`" for col in columns)

    # Multi-column layout
    result = []
    for i in range(0, len(columns), 3):
        group = columns[i : i + 3]
        row = " | ".join(f"`{col}`" for col in group)
        result.append(f"• {row}")
    return "\n".join(result)


def get_type_guidance(expected_type: str, invalid_value: str) -> str:
    """Get specific guidance for type mismatches."""

    if expected_type in ["integer", "bigint", "int", "int4", "int8"]:
        return (
            "**Fix:** Remove quotes from numbers\n"
            f'❌ Wrong: `"value": "{invalid_value}"`\n'
            f'✅ Right: `"value": {invalid_value}` (if it\'s a valid number)\n\n'
            "If the value isn't a number, you may be using the wrong column."
        )

    elif expected_type in ["boolean", "bool"]:
        return (
            "**Fix:** Use boolean values without quotes\n"
            f'❌ Wrong: `"value": "{invalid_value}"`\n'
            '✅ Right: `"value": true` or `"value": false`'
        )

    elif expected_type in ["timestamp", "timestamptz", "date"]:
        return (
            "**Fix:** Use proper timestamp format\n"
            f"Current value: `{invalid_value}`\n"
            "✅ Valid formats:\n"
            '• ISO 8601: `"2024-01-15T10:30:00Z"`\n'
            '• Date only: `"2024-01-15"`\n'
            '• PostgreSQL: `"2024-01-15 10:30:00"`'
        )

    elif expected_type == "uuid":
        return (
            "**Fix:** Provide a valid UUID\n"
            f"Current value: `{invalid_value}`\n"
            '✅ Valid UUID format: `"550e8400-e29b-41d4-a716-446655440000"`\n\n'
            "If you don't have a UUID, you might be using the wrong column."
        )

    elif expected_type in ["json", "jsonb"]:
        return (
            "**Fix:** Provide valid JSON\n"
            f"Current value: `{invalid_value}`\n"
            "For JSONB queries, use appropriate operators:\n"
            "• `@>` for contains\n"
            "• `->>` for text extraction\n"
            "• `->` for JSON extraction"
        )

    else:
        return (
            f"**Fix:** Ensure the value matches the expected type `{expected_type}`\n"
            f"Current value: `{invalid_value}`"
        )


def find_similar_columns(
    column: str, valid_columns: List[str], threshold: float = 0.6
) -> List[str]:
    """
    Find similar column names using simple string similarity.

    Args:
        column: The incorrect column name
        valid_columns: List of valid column names
        threshold: Similarity threshold (0-1)

    Returns:
        List of similar column names
    """
    from difflib import SequenceMatcher

    similar = []
    column_lower = column.lower()

    for valid_col in valid_columns:
        # Check exact match (case-insensitive)
        if valid_col.lower() == column_lower:
            return [valid_col]  # Exact match found

        # Calculate similarity
        similarity = SequenceMatcher(None, column_lower, valid_col.lower()).ratio()
        if similarity >= threshold:
            similar.append((valid_col, similarity))

    # Sort by similarity and return top 3
    similar.sort(key=lambda x: x[1], reverse=True)
    return [col for col, _ in similar[:3]]


async def get_table_columns(supabase_client, table: str) -> List[str]:
    """
    Get column names for a table from table_metadata.

    Args:
        supabase_client: Supabase client instance
        table: Table name

    Returns:
        List of column names
    """
    global _table_schemas_cache

    # Check cache first (with thread-safe access)
    with _cache_lock:
        if table in _table_schemas_cache:
            return _table_schemas_cache[table]

    try:
        # Query table_metadata for column information
        result = (
            supabase_client.table("table_metadata")
            .select("column_summary")
            .eq("table_name", table)
            .execute()
        )

        if result.data and len(result.data) > 0:
            column_summary = result.data[0].get("column_summary", "")
            # Parse column names from summary
            columns = []
            if column_summary:
                # Column summary format: "column1, column2, column3"
                columns = [col.strip() for col in column_summary.split(",")]

            # Cache the result (with thread-safe access)
            with _cache_lock:
                _table_schemas_cache[table] = columns
            return columns
    except Exception as e:
        logger.warning(f"Failed to get columns for table {table}: {e}")

    return []
