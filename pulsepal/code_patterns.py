"""
Function clustering module for PulsePal.

This module provides function clustering - identifying which Pulseq functions
commonly work together. This is valuable because:
1. Pulseq has a finite set of ~150 functions that rarely changes
2. The relationships between functions are domain-specific knowledge
3. It's lightweight and doesn't require constant updates

"""

from typing import List, Dict, Set, Tuple
from .function_index import FUNCTION_CLUSTERS


class FunctionClusterAnalyzer:
    """
    Analyzes which Pulseq functions are commonly used together.
    This helps identify potentially missing functions in user code.
    """

    def __init__(self):
        # Load function clusters from function_index
        self.function_clusters = FUNCTION_CLUSTERS

        # Build reverse mapping for quick lookups
        self.function_to_clusters = self._build_reverse_mapping()

    def _build_reverse_mapping(self) -> Dict[str, List[str]]:
        """Build reverse mapping from function to clusters."""
        function_to_clusters = {}
        for cluster_name, functions in self.function_clusters.items():
            for func in functions:
                if func not in function_to_clusters:
                    function_to_clusters[func] = []
                function_to_clusters[func].append(cluster_name)
        return function_to_clusters

    def analyze_functions(self, used_functions: List[str]) -> Dict:
        """
        Analyze which function clusters are being used and what might be missing.

        Args:
            used_functions: List of Pulseq functions found in the code

        Returns:
            Dictionary containing:
            - clusters_detected: Which function clusters are in use
            - related_functions: All functions from detected clusters
            - potentially_missing: Functions that are often used with the current ones
        """
        # Clean function names (remove prefixes)
        cleaned_functions = self._clean_function_names(used_functions)

        # Detect which clusters are being used
        clusters_detected = self._detect_clusters(cleaned_functions)

        # Get all related functions from those clusters
        related_functions = self._get_related_functions(clusters_detected)

        # Find potentially missing functions
        potentially_missing = related_functions - cleaned_functions

        return {
            "clusters_detected": list(clusters_detected),
            "related_functions": list(related_functions),
            "potentially_missing": list(potentially_missing),
            "functions_analyzed": list(cleaned_functions),
        }

    def _clean_function_names(self, functions: List[str]) -> Set[str]:
        """Remove prefixes from function names for matching."""
        cleaned = set()
        for func in functions:
            if func:
                # Remove common prefixes
                clean = func.replace("mr.", "").replace("seq.", "").replace("opt.", "")
                # Also handle eve.*, tra.*, mr.aux.* patterns
                if "." in clean:
                    # Take the last part after any remaining dots
                    clean = clean.split(".")[-1]
                cleaned.add(clean)
        return cleaned

    def _detect_clusters(self, functions: Set[str]) -> Set[str]:
        """Detect which function clusters are being used."""
        clusters = set()
        for func in functions:
            if func in self.function_to_clusters:
                clusters.update(self.function_to_clusters[func])
        return clusters

    def _get_related_functions(self, clusters: Set[str]) -> Set[str]:
        """Get all functions from the detected clusters."""
        related = set()
        for cluster_name in clusters:
            related.update(self.function_clusters[cluster_name])
        return related

    def get_cluster_info(self, cluster_name: str) -> Dict:
        """
        Get information about a specific cluster.

        Args:
            cluster_name: Name of the cluster

        Returns:
            Dictionary with cluster information
        """
        if cluster_name not in self.function_clusters:
            return {"exists": False, "functions": []}

        return {
            "exists": True,
            "name": cluster_name,
            "functions": self.function_clusters[cluster_name],
            "description": self._get_cluster_description(cluster_name),
        }

    def _get_cluster_description(self, cluster_name: str) -> str:
        """Get a human-readable description of what a cluster is for."""
        descriptions = {
            "rf_excitation": "RF pulse generation and excitation",
            "slice_selection": "Slice-selective excitation with gradients",
            "readout_pattern": "Data acquisition and frequency encoding",
            "phase_encoding": "Phase encoding for spatial localization",
            "spoiling": "Gradient and RF spoiling for steady-state",
            "epi_readout": "Echo-planar imaging readout pattern",
            "diffusion_weighting": "Diffusion-weighted imaging gradients",
            "fat_saturation": "Fat suppression pulses",
            "flow_compensation": "Flow and motion compensation",
            "calibration": "Calibration and adjustment scans",
        }
        return descriptions.get(cluster_name, f"Function cluster: {cluster_name}")

    def find_missing_core_functions(
        self, used_functions: List[str], clusters: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Find core functions that are missing from specific clusters.
        This is more targeted than potentially_missing.

        Args:
            used_functions: Functions currently in use
            clusters: Clusters detected in the code

        Returns:
            List of (function, reason) tuples for important missing functions
        """
        cleaned_functions = self._clean_function_names(used_functions)
        missing_core = []

        # Define core functions for each cluster
        # These are functions that are almost always needed when using the cluster
        core_functions = {
            "rf_excitation": {
                "makeSincPulse": "Needed for slice-selective excitation",
                "makeBlockPulse": "Alternative for non-selective excitation",
            },
            "readout_pattern": {
                "makeAdc": "Required for data acquisition",
                "makeTrapezoid": "Required for readout gradient",
            },
            "phase_encoding": {
                "makeTrapezoid": "Required for phase encoding gradients"
            },
            "slice_selection": {
                "makeTrapezoid": "Required for slice selection gradient"
            },
        }

        for cluster in clusters:
            if cluster in core_functions:
                for func, reason in core_functions[cluster].items():
                    if func not in cleaned_functions:
                        # Check if at least one of the alternatives is present
                        if cluster == "rf_excitation":
                            # For RF, either sincPulse OR blockPulse is fine
                            if (
                                "makeSincPulse" not in cleaned_functions
                                and "makeBlockPulse" not in cleaned_functions
                            ):
                                missing_core.append((func, reason))
                        else:
                            missing_core.append((func, reason))

        return missing_core
