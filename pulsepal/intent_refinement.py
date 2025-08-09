"""
Phase 4: Understand intent and refine code with corrections.
"""
from typing import Dict, List, Optional

class IntentRefinement:
    def __init__(self):
        self.MISSING_FUNCTIONALITY = {
            'sar_calculation': {
                'intent': 'Calculate SAR',
                'alternative': '% Calculate manually: SAR ∝ flip_angle² × RF_duration / TR'
            },
            'snr_calculation': {
                'intent': 'Calculate SNR',
                'alternative': '% SNR requires image data, not part of sequence design'
            },
            'coil_sensitivity': {
                'intent': 'Handle coil sensitivity',
                'alternative': '% Handled in reconstruction, not sequence design'
            },
        }
    
    def identify_intent(self, hallucinated_functions: List[str], code_context: str) -> Dict:
        """Identify what user is trying to achieve"""
        intent = {
            'primary_goal': None,
            'missing_capabilities': [],
            'alternatives': []
        }
        
        for hallucinated in hallucinated_functions:
            lower_hall = hallucinated.lower()
            
            if 'sar' in lower_hall:
                intent['missing_capabilities'].append('SAR calculation')
                intent['alternatives'].append(self.MISSING_FUNCTIONALITY['sar_calculation'])
            elif 'snr' in lower_hall or 'noise' in lower_hall:
                intent['missing_capabilities'].append('SNR calculation')
                intent['alternatives'].append(self.MISSING_FUNCTIONALITY['snr_calculation'])
            elif 'coil' in lower_hall:
                intent['missing_capabilities'].append('Coil sensitivity')
                intent['alternatives'].append(self.MISSING_FUNCTIONALITY['coil_sensitivity'])
        
        # Identify primary goal from context
        if 'makeSincPulse' in code_context and 'makeTrapezoid' in code_context:
            intent['primary_goal'] = 'Creating excitation with slice selection'
        elif 'makeAdc' in code_context:
            intent['primary_goal'] = 'Setting up data acquisition'
        elif 'makeBlockPulse' in code_context:
            intent['primary_goal'] = 'Creating refocusing pulse'
        elif 'EPI' in code_context or 'epi' in code_context:
            intent['primary_goal'] = 'Creating EPI readout'
        elif 'diffusion' in code_context.lower():
            intent['primary_goal'] = 'Adding diffusion weighting'
        
        return intent
    
    def add_warning_comments(self, code: str, missing_capabilities: List[str]) -> str:
        """Add warning comments for missing functionality"""
        if not missing_capabilities:
            return code
        
        warning = "% WARNING: The following functionality is not available in Pulseq:\n"
        for capability in missing_capabilities:
            warning += f"% - {capability}\n"
            # Add alternative approach if available
            for func_info in self.MISSING_FUNCTIONALITY.values():
                if func_info['intent'] == capability:
                    warning += f"%   Alternative: {func_info['alternative']}\n"
        
        return warning + "\n" + code