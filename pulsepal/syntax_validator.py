"""
Syntax validation for Pulseq function calls.
Repurposed from function_similarity.py for debugging.
This provides deterministic validation based on the function index.
"""

from typing import Dict, List, Optional

from .function_index import MATLAB_FUNCTIONS


class SyntaxValidator:
    """Validates Pulseq function syntax and suggests corrections."""

    def __init__(self):
        self.direct_calls = MATLAB_FUNCTIONS.get("direct_calls", set())
        self.class_methods = MATLAB_FUNCTIONS.get("class_methods", {})
        self.all_functions = set()

        # Build complete function list
        # direct_calls is a set of function names
        self.all_functions.update(self.direct_calls)

        # class_methods is a dictionary with class names as keys
        if isinstance(self.class_methods, dict):
            for methods in self.class_methods.values():
                if isinstance(methods, (set, list)):
                    self.all_functions.update(methods)

    def validate_function_call(self, function_call: str) -> Dict:
        """
        Validate a single function call.

        Args:
            function_call: String like 'mr.makeTrapezoid' or 'seq.write'

        Returns:
            Dictionary with:
            - is_valid: Boolean
            - correct_form: Correct syntax if invalid
            - explanation: Why it's wrong
        """
        # Parse the function call
        if "." in function_call:
            namespace, method = function_call.split(".", 1)

            # Check if it's a valid namespace
            if namespace == "mr":
                # Check if it's a valid mr function (from direct_calls)
                if method in self.direct_calls:
                    return {"is_valid": True}
                # Check if it's actually a sequence method
                if method in self.class_methods.get("seq", []):
                    return {
                        "is_valid": False,
                        "correct_form": f"seq.{method}",
                        "explanation": f"{method} is a sequence object method, not an mr function",
                    }

            elif namespace == "seq":
                # Check if it's a valid sequence method
                if method in self.class_methods.get("seq", []):
                    return {"is_valid": True}
                # Check if it's actually an mr function
                if method in self.direct_calls:
                    return {
                        "is_valid": False,
                        "correct_form": f"mr.{method}",
                        "explanation": f"{method} is an mr function, not a sequence method",
                    }

        # Check standalone functions (these are in direct_calls)
        if function_call in self.direct_calls:
            return {"is_valid": True}

        # Function not found - suggest similar
        suggestion = self.find_similar_function(function_call)
        if suggestion:
            return {
                "is_valid": False,
                "correct_form": suggestion,
                "explanation": f"Did you mean {suggestion}?",
            }

        return {
            "is_valid": False,
            "correct_form": None,
            "explanation": f"Function {function_call} not found in Pulseq",
        }

    def find_similar_function(self, incorrect_name: str) -> Optional[str]:
        """
        Find similar valid function names for typos.
        Uses edit distance and pattern matching.
        """
        # Remove namespace if present for matching
        base_name = (
            incorrect_name.split(".")[-1] if "." in incorrect_name else incorrect_name
        )

        # Look for exact matches with different namespace
        for func in self.all_functions:
            if func.endswith(base_name):
                return func

        # Look for partial matches
        for func in self.all_functions:
            if base_name.lower() in func.lower() or func.lower() in base_name.lower():
                return func

        return None

    def check_parameters(self, function_name: str, provided_params: List[str]) -> Dict:
        """
        Check if required parameters are provided.

        Returns:
            Dictionary with missing/extra parameters
        """
        # Basic parameter checking for common functions
        # This could be expanded with database lookups
        required_params = {
            "mr.makeTrapezoid": ["channel"],  # First param is always channel
            "mr.makeSincPulse": ["flip_angle"],
            "seq.addBlock": [],  # Variable parameters
            "seq.write": ["filename"],
            "mr.makeAdc": ["num_samples"],
            "mr.makeDelay": ["delay_time"],
        }

        if function_name in required_params:
            required = required_params[function_name]
            missing = [p for p in required if p not in provided_params]
            return {"missing_parameters": missing, "is_valid": len(missing) == 0}

        return {"is_valid": True, "missing_parameters": []}
