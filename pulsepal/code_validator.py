"""
AST-based code validator for PulsePal.
Validates MATLAB and Python code for correct Pulseq function usage.
"""

import re
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from .function_index import MATLAB_FUNCTIONS, COMMON_HALLUCINATIONS

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of code validation."""

    is_valid: bool
    errors: List[Dict[str, str]]  # List of {line, function, error, suggestion}
    warnings: List[Dict[str, str]]
    fixed_code: Optional[str] = None
    summary: Optional[str] = None


class PulseqCodeValidator:
    """
    Validates MATLAB/Python code for correct Pulseq function usage.
    Uses pattern matching and function index validation.
    """

    def __init__(self):
        """Initialize validator with function index."""
        self.all_functions = self._build_function_set()
        self.namespace_map = self._build_namespace_map()

    def _build_function_set(self) -> set:
        """Build set of all valid function names."""
        functions = set()

        # Add direct calls
        functions.update(MATLAB_FUNCTIONS.get("direct_calls", set()))

        # Add class methods
        for class_name, methods in MATLAB_FUNCTIONS.get("class_methods", {}).items():
            functions.update(methods)

        return functions

    def _build_namespace_map(self) -> Dict[str, str]:
        """Build map of function to correct namespace."""
        namespace_map = {}

        # Direct calls typically use 'mr.'
        for func in MATLAB_FUNCTIONS.get("direct_calls", set()):
            # Special cases
            if func == "Sequence":
                namespace_map[func] = ""  # Constructor
            elif func == "opts":
                namespace_map[func] = "mr"
            elif func.startswith("make"):
                namespace_map[func] = "mr"
            else:
                namespace_map[func] = "mr"

        # Class methods
        for class_name, methods in MATLAB_FUNCTIONS.get("class_methods", {}).items():
            for method in methods:
                if class_name == "Sequence":
                    namespace_map[method] = "seq"  # seq.method()
                elif class_name == "SeqPlot":
                    namespace_map[method] = "plot"

        return namespace_map

    def validate_code(self, code: str, language: str = "matlab") -> ValidationResult:
        """
        Validate code for correct Pulseq function usage.

        Args:
            code: Code to validate
            language: Language of code ('matlab' or 'python')

        Returns:
            ValidationResult with errors, warnings, and optionally fixed code
        """
        errors = []
        warnings = []
        lines = code.split("\n")
        fixed_lines = []

        for line_num, line in enumerate(lines, 1):
            # Skip comments
            if language == "matlab" and line.strip().startswith("%"):
                fixed_lines.append(line)
                continue
            elif language == "python" and line.strip().startswith("#"):
                fixed_lines.append(line)
                continue

            # Extract function calls from line
            function_calls = self._extract_function_calls(line, language)
            fixed_line = line

            for call in function_calls:
                validation = self._validate_function_call(call, language)

                if validation["error"]:
                    errors.append(
                        {
                            "line": line_num,
                            "function": call,
                            "error": validation["error"],
                            "suggestion": validation.get("suggestion", ""),
                        }
                    )

                    # Apply fix if available
                    if validation.get("fixed"):
                        fixed_line = fixed_line.replace(call, validation["fixed"])

                elif validation.get("warning"):
                    warnings.append(
                        {
                            "line": line_num,
                            "function": call,
                            "warning": validation["warning"],
                        }
                    )

            fixed_lines.append(fixed_line)

        # Build summary
        summary = self._build_summary(errors, warnings)

        # Only provide fixed code if there were errors that could be fixed
        fixed_code = None
        if errors and any(e.get("suggestion") for e in errors):
            fixed_code = "\n".join(fixed_lines)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            fixed_code=fixed_code,
            summary=summary,
        )

    def _extract_function_calls(self, line: str, language: str) -> List[str]:
        """Extract function calls from a line of code."""
        calls = []

        if language == "matlab":
            # MATLAB patterns
            patterns = [
                r"\b(mr|seq|tra|eve|opt)\.([A-Za-z]\w*)",  # namespace.function
                r"\b(mr)\.aux\.([A-Za-z]\w*)",  # mr.aux.function
                r"\b(mr)\.aux\.quat\.([A-Za-z]\w*)",  # mr.aux.quat.function
                r"\b([A-Z][A-Za-z]+)\s*\(",  # Constructor calls like Sequence()
            ]
        else:  # python
            patterns = [
                r"\b(mr|seq|tra|eve|opt)\.([A-Za-z]\w*)",
                r"\bpypulseq\.([A-Za-z]\w*)",
                r"\b([A-Z][A-Za-z]+)\s*\(",
            ]

        for pattern in patterns:
            matches = re.finditer(pattern, line)
            for match in matches:
                calls.append(match.group(0).rstrip("("))

        return calls

    def _validate_function_call(self, call: str, language: str) -> Dict:
        """
        Validate a single function call.

        Returns dict with:
        - error: Error message if invalid
        - suggestion: Suggested fix
        - fixed: Fixed function call
        - warning: Warning message (for valid but questionable usage)
        """
        result = {"error": None, "suggestion": None, "fixed": None, "warning": None}

        # Parse the call
        parts = call.split(".")

        if len(parts) == 1:
            # Direct function or constructor
            func_name = parts[0]

            # Check case sensitivity for known functions
            if func_name.lower() in [f.lower() for f in self.all_functions]:
                # Find correct case
                for correct_func in self.all_functions:
                    if correct_func.lower() == func_name.lower():
                        if correct_func != func_name:
                            result["error"] = f"Incorrect capitalization: '{func_name}'"
                            result["suggestion"] = f"Use '{correct_func}'"
                            result["fixed"] = correct_func
                        break
            elif func_name in COMMON_HALLUCINATIONS:
                correct = COMMON_HALLUCINATIONS[func_name]
                if correct:
                    result["error"] = f"'{func_name}' is not a valid Pulseq function"
                    result["suggestion"] = f"Use '{correct}'"
                    result["fixed"] = (
                        f"mr.{correct}"
                        if correct in MATLAB_FUNCTIONS.get("direct_calls", set())
                        else correct
                    )
                else:
                    result["error"] = f"'{func_name}' does not exist in Pulseq"

        elif len(parts) >= 2:
            # Namespace.function call
            namespace = parts[0]
            func_name = parts[-1]  # Last part is the function

            # Special case: mr.Opts should be mr.opts
            if namespace == "mr" and func_name == "Opts":
                result["error"] = "Incorrect capitalization: 'mr.Opts'"
                result["suggestion"] = "Use 'mr.opts'"
                result["fixed"] = "mr.opts"
                return result

            # Check if function exists (case-insensitive first)
            func_lower = func_name.lower()
            found_func = None

            for valid_func in self.all_functions:
                if valid_func.lower() == func_lower:
                    found_func = valid_func
                    break

            if found_func:
                # Check capitalization
                if found_func != func_name:
                    result["error"] = f"Incorrect capitalization: '{func_name}'"
                    result["suggestion"] = f"Use '{found_func}'"
                    result["fixed"] = f"{namespace}.{found_func}"

                # Check namespace
                elif found_func in self.namespace_map:
                    correct_ns = self.namespace_map[found_func]
                    if correct_ns and namespace != correct_ns:
                        result["error"] = f"Wrong namespace for '{func_name}'"
                        result["suggestion"] = f"Use '{correct_ns}.{func_name}'"
                        result["fixed"] = f"{correct_ns}.{func_name}"

            elif func_name in COMMON_HALLUCINATIONS:
                correct = COMMON_HALLUCINATIONS[func_name]
                if correct:
                    # Determine correct namespace
                    if correct in self.namespace_map:
                        correct_ns = self.namespace_map[correct]
                        result["error"] = f"'{func_name}' is not valid"
                        result["suggestion"] = f"Use '{correct_ns}.{correct}'"
                        result["fixed"] = f"{correct_ns}.{correct}"
                else:
                    result["error"] = f"'{func_name}' does not exist in Pulseq"
            else:
                # Function not found at all
                result["error"] = f"Unknown function: '{func_name}'"

        return result

    def _build_summary(self, errors: List[Dict], warnings: List[Dict]) -> str:
        """Build a summary of validation results."""
        if not errors and not warnings:
            return "✅ Code validation passed - all Pulseq functions are correct"

        parts = []

        if errors:
            parts.append(f"❌ Found {len(errors)} error(s):")
            for err in errors[:3]:  # Show first 3 errors
                parts.append(f"  • Line {err['line']}: {err['error']}")
                if err.get("suggestion"):
                    parts.append(f"    → {err['suggestion']}")
            if len(errors) > 3:
                parts.append(f"  ... and {len(errors) - 3} more")

        if warnings:
            parts.append(f"⚠️ Found {len(warnings)} warning(s)")

        return "\n".join(parts)


def validate_matlab_code(code: str) -> ValidationResult:
    """Convenience function to validate MATLAB code."""
    validator = PulseqCodeValidator()
    return validator.validate_code(code, "matlab")


def validate_python_code(code: str) -> ValidationResult:
    """Convenience function to validate Python code."""
    validator = PulseqCodeValidator()
    return validator.validate_code(code, "python")
