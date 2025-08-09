"""
Main module for preventing Pulseq function hallucinations in Gemini responses.
Implements 4-phase progressive grounding approach.
"""
import re
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from .function_index import MATLAB_FUNCTIONS, COMMON_HALLUCINATIONS

@dataclass
class HallucinationMetrics:
    """Metrics for tracking performance"""
    phase1_time: float = 0.0
    phase2_time: float = 0.0
    phase3_time: float = 0.0
    phase4_time: float = 0.0
    hallucinations_found: List[str] = field(default_factory=list)
    corrections_made: Dict[str, str] = field(default_factory=dict)
    success: bool = False

class PulseqGrounder:
    """
    Prevents hallucinations in Pulseq code generation.
    4-phase approach: Verify -> Find Similar -> Get Context -> Refine
    """
    
    def __init__(self):
        # Initialize all phases
        from .function_similarity import FunctionSimilarity
        from .function_clustering import FunctionClustering
        from .intent_refinement import IntentRefinement
        
        self.phase2 = FunctionSimilarity()
        self.phase3 = FunctionClustering()
        self.phase4 = IntentRefinement()
        
        # Quick lookup sets for Phase 1
        self.direct_calls = MATLAB_FUNCTIONS['direct_calls']
        self.class_methods = MATLAB_FUNCTIONS['class_methods']
    
    def extract_function_calls(self, code: str) -> List[Tuple[str, str]]:
        """
        Extract all function calls from MATLAB code.
        Returns list of (function_name, full_match) tuples.
        """
        patterns = [
            (r'mr\.(\w+)\s*\(', 'mr.{}'),
            (r'seq\.(\w+)\s*\(', 'seq.{}'),
            (r'=\s*(make\w+)\s*\(', '{}'),
            (r'^\s*(make\w+)\s*\(', '{}'),
            (r',\s*(make\w+)\s*\(', '{}'),
            (r'\[\w+,?\s*\w*\]\s*=\s*mr\.(\w+)\s*\(', 'mr.{}'),
        ]
        
        function_calls = []
        for pattern, format_str in patterns:
            matches = re.finditer(pattern, code, re.MULTILINE)
            for match in matches:
                func_name = match.group(1)
                full_name = format_str.format(func_name)
                function_calls.append((func_name, full_name))
        
        return function_calls
    
    def is_valid_function(self, func_name: str, full_name: str) -> bool:
        """Phase 1: Check if function exists"""
        # Check direct calls
        if func_name in self.direct_calls:
            return True
        
        # Check class methods
        if 'seq.' in full_name:
            if func_name in self.class_methods.get('Sequence', set()):
                return True
        
        # Check other class methods
        for class_name, methods in self.class_methods.items():
            if func_name in methods:
                return True
        
        return False
    
    async def prevent_hallucination(self, code: str) -> Dict:
        """
        Main entry point for hallucination prevention.
        """
        metrics = HallucinationMetrics()
        
        # Phase 1: Verification (<5ms)
        t1 = time.time()
        function_calls = self.extract_function_calls(code)
        hallucinated = []
        valid = []
        
        for func_name, full_name in function_calls:
            if self.is_valid_function(func_name, full_name):
                valid.append(full_name)
            else:
                hallucinated.append(full_name)
        
        metrics.phase1_time = (time.time() - t1) * 1000
        metrics.hallucinations_found = hallucinated
        
        if not hallucinated:
            metrics.success = True
            return {
                'code': code,
                'metrics': metrics,
                'corrections': {},
                'modified': False
            }
        
        # Phase 2: Find similar functions (<20ms)
        t2 = time.time()
        corrections = {}
        for hall_func in hallucinated:
            similar = self.phase2.find_similar(hall_func, code)
            if similar:
                corrections[hall_func] = similar[0][0]
            else:
                corrections[hall_func] = None
        
        metrics.phase2_time = (time.time() - t2) * 1000
        metrics.corrections_made = corrections
        
        # Phase 3: Get context (<50ms)
        t3 = time.time()
        cluster_context = self.phase3.get_related_functions(
            valid, 
            [c for c in corrections.values() if c]
        )
        metrics.phase3_time = (time.time() - t3) * 1000
        
        # Phase 4: Refine (<200ms)
        t4 = time.time()
        intent = self.phase4.identify_intent(hallucinated, code)
        
        # For now, use simple replacement (Gemini integration later)
        refined_code = self.simple_refinement(code, corrections)
        
        if intent.get('missing_capabilities'):
            refined_code = self.phase4.add_warning_comments(
                refined_code, 
                intent['missing_capabilities']
            )
        
        metrics.phase4_time = (time.time() - t4) * 1000
        metrics.success = True
        
        return {
            'code': refined_code,
            'metrics': metrics,
            'corrections': corrections,
            'modified': True,
            'context': {**cluster_context, 'intent': intent}
        }
    
    def simple_refinement(self, code: str, corrections: Dict[str, Optional[str]]) -> str:
        """Simple replacement without Gemini"""
        refined = code
        for wrong, right in corrections.items():
            if right:
                # Handle both standalone and mr. prefix cases
                if 'mr.' in wrong:
                    refined = refined.replace(wrong, f"mr.{right}" if not right.startswith('seq.') else right)
                else:
                    refined = refined.replace(wrong, right)
            else:
                refined = refined.replace(
                    wrong,
                    f"{wrong} % WARNING: Function doesn't exist in Pulseq"
                )
        return refined