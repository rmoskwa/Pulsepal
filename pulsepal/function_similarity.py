"""
Phase 2: Find similar functions when hallucination detected.
"""
import difflib
from typing import List, Tuple, Optional

class FunctionSimilarity:
    def __init__(self):
        self.PREFIXES = {
            'make': 'Creates/constructs',
            'calc': 'Calculates',
            'set': 'Sets parameter',
            'get': 'Retrieves value',
            'add': 'Adds to sequence',
            'register': 'Registers event',
        }
        
        self.OBJECT_TYPES = {
            'Pulse': ['Sinc', 'Gauss', 'Block', 'Adiabatic', 'SLR'],
            'Gradient': ['Trapezoid', 'Arbitrary', 'Extended'],
            'Delay': ['Delay', 'SoftDelay'],
            'ADC': ['Adc'],
            'RF': ['Rf', 'ArbitraryRf'],
        }
    
    def find_similar(self, hallucinated: str, context: str = None) -> List[Tuple[str, float]]:
        """Find similar functions with confidence scores"""
        from .function_index import COMMON_HALLUCINATIONS, MATLAB_FUNCTIONS
        
        # Clean the hallucinated function name
        hallucinated_clean = hallucinated.replace('mr.', '').replace('seq.', '')
        
        # Check common mistakes first
        if hallucinated_clean in COMMON_HALLUCINATIONS:
            correct = COMMON_HALLUCINATIONS[hallucinated_clean]
            if correct:
                return [(correct, 1.0)]
            else:
                return []
        
        # Analyze structure
        suggestions = []
        for prefix in self.PREFIXES:
            if hallucinated_clean.lower().startswith(prefix.lower()):
                obj_part = hallucinated_clean[len(prefix):]
                suggestions.extend(self._find_by_object_type(prefix, obj_part))
        
        # Fuzzy matching
        all_functions = list(MATLAB_FUNCTIONS['direct_calls'])
        for class_name, methods in MATLAB_FUNCTIONS['class_methods'].items():
            for method in methods:
                if class_name == 'Sequence':
                    all_functions.append(f"seq.{method}")
                else:
                    all_functions.append(method)
        
        close_matches = difflib.get_close_matches(
            hallucinated_clean, 
            all_functions, 
            n=5, 
            cutoff=0.6
        )
        
        for match in close_matches:
            score = difflib.SequenceMatcher(None, hallucinated_clean, match).ratio()
            suggestions.append((match, score))
        
        # Remove duplicates and sort
        seen = set()
        unique = []
        for func, score in sorted(suggestions, key=lambda x: x[1], reverse=True):
            if func not in seen:
                seen.add(func)
                unique.append((func, score))
        
        return unique[:5]
    
    def _find_by_object_type(self, prefix: str, obj_part: str) -> List[Tuple[str, float]]:
        """Find functions based on prefix and object type"""
        suggestions = []
        from .function_index import MATLAB_FUNCTIONS
        
        for category, variants in self.OBJECT_TYPES.items():
            for variant in variants:
                if variant.lower() in obj_part.lower():
                    if category == 'Pulse':
                        func_name = f"make{variant}Pulse"
                    elif category == 'Gradient' and variant == 'Trapezoid':
                        func_name = 'makeTrapezoid'
                    elif category == 'ADC':
                        func_name = 'makeAdc'
                    else:
                        func_name = f"{prefix}{variant}"
                    
                    if func_name in MATLAB_FUNCTIONS['direct_calls']:
                        suggestions.append((func_name, 0.8))
        
        return suggestions