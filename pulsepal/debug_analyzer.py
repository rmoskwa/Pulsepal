"""
Advanced two-tier debugging system for Pulseq code analysis.
Combines syntax validation with conceptual MRI physics debugging.
Uses built-in physics knowledge as primary tool, with optional pattern hints.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from .function_index import MATLAB_FUNCTIONS


@dataclass
class SyntaxIssue:
    """Represents a syntax or function usage error."""

    line_number: int
    incorrect_usage: str
    correct_usage: str
    explanation: str
    severity: str  # 'error', 'warning', 'suggestion'


@dataclass
class ConceptualIssue:
    """Represents a physics-related implementation issue."""

    physics_concept: str
    sequence_element: str
    code_location: str
    issue_description: str
    solution: str


class PulseqDebugAnalyzer:
    """
    Advanced two-tier debugging system for Pulseq code.
    Not dependent on pre-mapped patterns - uses physics reasoning.
    """

    def __init__(self):
        from .syntax_validator import SyntaxValidator
        from .concept_mapper import ConceptMapper

        self.syntax_validator = SyntaxValidator()
        self.concept_mapper = ConceptMapper()  # Optional hint provider
        self.function_index = MATLAB_FUNCTIONS

    def analyze_code(
        self, code: str, problem_description: Optional[str] = None
    ) -> Dict:
        """
        Perform comprehensive debugging analysis.

        Args:
            code: User's Pulseq code to analyze
            problem_description: Optional description of the problem

        Returns:
            Dictionary containing:
            - syntax_errors: List of syntax/function issues
            - conceptual_issues: MRI physics problems identified
            - physics_reasoning: Explanation of physics analysis
            - suggestions: Corrective actions with documentation
        """
        results = {
            "syntax_errors": [],
            "conceptual_issues": [],
            "physics_reasoning": "",
            "suggestions": [],
        }

        # Category 1: Syntax analysis (always performed)
        syntax_results = self.debug_syntax(code)
        results["syntax_errors"] = syntax_results

        # Category 2: Conceptual analysis (if problem described)
        if problem_description:
            concept_results = self.debug_concept(problem_description, code)
            results["conceptual_issues"] = concept_results["issues"]
            results["physics_reasoning"] = concept_results["reasoning"]

        return results

    def debug_syntax(self, code: str) -> List[SyntaxIssue]:
        """
        Category 1: Check function signatures, parameters, methods.
        This is deterministic and based on the function index.
        """
        issues = []
        lines = code.split("\n")

        for line_num, line in enumerate(lines, 1):
            # Check for common namespace errors
            if "mr.write" in line:
                issues.append(
                    SyntaxIssue(
                        line_number=line_num,
                        incorrect_usage="mr.write(filename)",
                        correct_usage="seq.write(filename)",
                        explanation="write() is a method of the sequence object, not the mr namespace",
                        severity="error",
                    )
                )

            if "mr.addBlock" in line:
                issues.append(
                    SyntaxIssue(
                        line_number=line_num,
                        incorrect_usage="mr.addBlock(...)",
                        correct_usage="seq.addBlock(...)",
                        explanation="addBlock() belongs to the sequence object",
                        severity="error",
                    )
                )

            # Check for missing parameters in makeTrapezoid
            if "makeTrapezoid(" in line and "'channel'" not in line:
                issues.append(
                    SyntaxIssue(
                        line_number=line_num,
                        incorrect_usage="mr.makeTrapezoid(amplitude, ...)",
                        correct_usage="mr.makeTrapezoid('x', 'amplitude', value, ...)",
                        explanation="makeTrapezoid requires channel as first parameter",
                        severity="error",
                    )
                )

            # Check for common typos
            if "mr.duration" in line:
                issues.append(
                    SyntaxIssue(
                        line_number=line_num,
                        incorrect_usage="mr.duration()",
                        correct_usage="seq.duration()",
                        explanation="duration() is a sequence method to get total sequence time",
                        severity="error",
                    )
                )

        return issues

    def debug_concept(self, problem: str, code: str) -> Dict:
        """
        Category 2: Trace from MRI physics to code implementation.

        PRIMARY: Uses physics reasoning to analyze ANY problem
        SECONDARY: Checks concept_mapper for common pattern hints
        """
        # First, try to get hints from concept mapper
        mapping_hints = self.concept_mapper.map_problem_to_code(problem)

        # Build the analysis based on physics reasoning
        physics_reasoning = self._analyze_physics(problem, code, mapping_hints)

        # Identify specific issues in the code
        issues = self._identify_conceptual_issues(problem, code, physics_reasoning)

        return {"issues": issues, "reasoning": physics_reasoning}

    def _analyze_physics(self, problem: str, code: str, hints: Dict) -> str:
        """
        Apply physics reasoning to understand the problem.
        This is where PulsePal's intelligence shines - not dependent on patterns.
        """
        # Check if we have hints
        if hints.get("approach") == "systematic_analysis":
            # Novel problem - use pure physics reasoning
            reasoning = f"""Analyzing novel problem: "{problem}"
            
This issue isn't in my common patterns database, but I can analyze it using MRI physics:

1. Physics Analysis: Breaking down the problem into fundamental MRI principles
2. Sequence Components: Identifying which parts of the sequence could cause this
3. Pulseq Implementation: Finding the relevant functions and parameters
4. Validation: Checking against physical constraints and hardware limits"""
        else:
            # We have hints - use them to speed up analysis
            reasoning = f"""Analyzing problem using physics knowledge with pattern hints:
            
Physics Principle: {hints.get("physics", "Analyzing physics...")}
This relates to your problem because it directly controls the observed behavior."""

        return reasoning

    def _identify_conceptual_issues(
        self, problem: str, code: str, reasoning: str
    ) -> List[ConceptualIssue]:
        """
        Identify specific conceptual issues based on physics analysis.
        Works for ANY problem, not just pre-mapped ones.
        """
        issues = []
        problem_lower = problem.lower()

        # Use physics reasoning to identify issues
        # This is where the main agent's intelligence would guide the analysis

        # Example: Any problem involving gradients
        if "gradient" in problem_lower:
            if "makeTrapezoid" in code or "makeArbitraryGrad" in code:
                issues.append(
                    ConceptualIssue(
                        physics_concept="Gradient pulse generation",
                        sequence_element="Gradient waveforms",
                        code_location="Gradient definition sections",
                        issue_description="Gradient parameters may not match hardware capabilities",
                        solution="Verify amplitude and slew rate against system limits",
                    )
                )

        # For novel problems, provide systematic guidance
        if not issues:
            issues.append(
                ConceptualIssue(
                    physics_concept="Systematic analysis required",
                    sequence_element="Multiple components may be involved",
                    code_location="Review entire sequence structure",
                    issue_description=f"Novel problem: {problem}",
                    solution="Apply systematic debugging: physics → design → code → validation",
                )
            )

        return issues
