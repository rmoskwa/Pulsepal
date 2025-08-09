"""
Phase 3: Retrieve related functions that commonly work together.
"""
from typing import List, Dict, Set

class FunctionClustering:
    def __init__(self):
        from .function_index import FUNCTION_CLUSTERS
        self.FUNCTION_CLUSTERS = FUNCTION_CLUSTERS
        self._build_reverse_mapping()
    
    def _build_reverse_mapping(self):
        """Build reverse mapping from function to clusters"""
        self.function_to_clusters = {}
        for cluster_name, functions in self.FUNCTION_CLUSTERS.items():
            for func in functions:
                if func not in self.function_to_clusters:
                    self.function_to_clusters[func] = []
                self.function_to_clusters[func].append(cluster_name)
    
    def get_related_functions(self, 
                            valid_functions: List[str], 
                            corrected_functions: List[str]) -> Dict:
        """Get context for the functions being used"""
        context = {
            'clusters_detected': [],
            'related_functions': set(),
            'usage_hints': [],
            'missing_functions': []
        }
        
        # Clean function names for lookup
        cleaned_functions = []
        for func in valid_functions + corrected_functions:
            if func:
                cleaned = func.replace('mr.', '').replace('seq.', '')
                cleaned_functions.append(cleaned)
        
        all_functions = set(cleaned_functions)
        
        # Find which clusters are being used
        clusters_used = set()
        for func in all_functions:
            if func in self.function_to_clusters:
                clusters_used.update(self.function_to_clusters[func])
        
        # Get all related functions
        for cluster_name in clusters_used:
            functions = self.FUNCTION_CLUSTERS[cluster_name]
            context['clusters_detected'].append(cluster_name)
            context['related_functions'].update(functions)
            
            # Find missing core functions
            missing = set(functions) - all_functions
            if missing:
                context['missing_functions'].extend(list(missing))
        
        # Add usage hints based on clusters
        if 'readout_pattern' in clusters_used:
            context['usage_hints'].append("Consider adding gradient prephasing before readout")
        if 'rf_excitation' in clusters_used:
            context['usage_hints'].append("Remember to include slice selection gradient with RF")
        if 'epi_readout' in clusters_used:
            context['usage_hints'].append("EPI requires gradient blips between readouts")
        if 'diffusion_weighting' in clusters_used:
            context['usage_hints'].append("Ensure b-value calculation matches gradient timing")
        
        context['related_functions'] = list(context['related_functions'])
        return context