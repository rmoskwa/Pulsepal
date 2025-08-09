"""
Maps MRI physics concepts to Pulseq code patterns for conceptual debugging.
This is a SUPPLEMENTARY tool that provides hints for common patterns.
For novel problems, the main agent uses its intelligence directly.

IMPORTANT: This tool is NOT required for PulsePal to function.
It's an optimization for common cases only.
"""

from typing import Dict


class ConceptMapper:
    """
    Provides hints for common MRI physics-to-code patterns.
    This is an optional performance optimization, not a requirement.
    """

    def __init__(self):
        self.concept_map = self._initialize_concept_map()

    def map_problem_to_code(self, problem_description: str) -> Dict:
        """
        Attempts to map problem to known patterns.
        Returns hints if found, or a generic framework if not.

        IMPORTANT: This is not exhaustive - novel problems are handled
        by the main agent's reasoning capabilities.

        Args:
            problem_description: User's description of the issue

        Returns:
            Dictionary with either:
            - Specific hints for known patterns
            - Generic framework for novel problems
        """
        problem_lower = problem_description.lower()

        # Try to find matching patterns
        for concept_key, concept_data in self.concept_map.items():
            # Check if any keywords match
            if any(
                keyword in problem_lower for keyword in concept_data.get("keywords", [])
            ):
                return concept_data

        # No match found - return generic framework
        return self._get_generic_debugging_framework(problem_description)

    def _get_generic_debugging_framework(self, problem: str) -> Dict:
        """
        Provides a generic framework for problems not in the map.
        This ensures PulsePal can handle ANY problem systematically.
        """
        return {
            "approach": "systematic_analysis",
            "physics_explanation": "Analyze using fundamental MRI physics principles",
            "responsible_elements": [
                "Identify all sequence components that could affect this"
            ],
            "check_functions": ["Examine relevant Pulseq functions based on physics"],
            "common_fixes": [
                "Verify physical parameters are within reasonable ranges",
                "Check timing relationships between sequence elements",
                "Validate against hardware limitations",
                "Ensure proper synchronization of components",
            ],
            "note": f'Novel problem detected: "{problem}". Using general physics reasoning.',
            "is_novel": True,
        }

    def _initialize_concept_map(self) -> Dict:
        """
        Initialize hints for COMMON patterns only.
        This is not meant to be exhaustive - just the most frequent issues.
        """
        return {
            "k-space_trajectory": {
                "keywords": ["k-space", "trajectory", "sampling", "center"],
                "physics": "Gradient waveforms control k-space traversal",
                "sequence_elements": [
                    "phase encoding",
                    "frequency encoding",
                    "readout",
                ],
                "pulseq_functions": ["mr.makeTrapezoid", "mr.makeArbitraryGrad"],
                "common_issues": [
                    "Incorrect gradient amplitudes",
                    "Wrong timing between gradients",
                    "Missing gradient balancing",
                    "Phase encoding not centered at k=0",
                ],
                "code_patterns": [
                    "delta_ky = 1/FOV",
                    "gy_area = (pe_step - Ny/2) * delta_ky",
                ],
                "is_novel": False,
            },
            "image_brightness": {
                "keywords": ["dark", "bright", "intensity", "signal", "contrast"],
                "physics": "Signal intensity depends on flip angle, TR, TE, and tissue parameters",
                "sequence_elements": ["RF pulses", "repetition time", "echo time"],
                "pulseq_functions": [
                    "mr.makeSincPulse",
                    "mr.makeBlockPulse",
                    "mr.makeDelay",
                ],
                "common_issues": [
                    "Flip angle not optimized for tissue contrast",
                    "TR too short for T1 recovery",
                    "TE too long causing excessive T2 decay",
                    "Ernst angle calculation incorrect",
                ],
                "is_novel": False,
            },
            "artifacts": {
                "keywords": [
                    "artifact",
                    "ghost",
                    "aliasing",
                    "distortion",
                    "chemical shift",
                ],
                "physics": "Various sources: aliasing, chemical shift, motion, eddy currents",
                "sequence_elements": ["FOV", "bandwidth", "timing", "gradients"],
                "pulseq_functions": [
                    "mr.makeAdc",
                    "mr.calcDuration",
                    "mr.makeTrapezoid",
                ],
                "common_issues": [
                    "FOV too small causing wraparound",
                    "Bandwidth too low for chemical shift",
                    "Gradient timing misalignment",
                    "Unbalanced gradients causing eddy currents",
                ],
                "is_novel": False,
            },
            # Only add VERY common patterns here
            # The system should work perfectly fine without any patterns
        }

    def add_pattern_from_experience(self, problem: str, solution: Dict) -> None:
        """
        Future enhancement: Learn from novel problems that get solved.
        This would allow the system to improve over time.
        """
        # This could log successful novel problem solutions
        # for future inclusion in the concept map
        pass
