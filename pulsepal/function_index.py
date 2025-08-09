"""
Hardcoded index of all 150 MATLAB Pulseq functions.
Generated from function_calling_patterns table in Supabase.
"""

MATLAB_FUNCTIONS = {
    'direct_calls': {
        # Core creation functions (most commonly used)
        'makeAdc', 'makeTrapezoid', 'makeDelay', 'makeSincPulse',
        'makeGaussPulse', 'makeBlockPulse', 'makeArbitraryGrad',
        'makeArbitraryRf', 'makeExtendedTrapezoid', 'makeExtendedTrapezoidArea',
        'makeSLRpulse', 'makeAdiabaticPulse', 'makeLabel', 'makeTrigger',
        'makeDigitalOutputPulse', 'makeSoftDelay',
        
        # Utility functions
        'opts', 'calcDuration', 'calcRfCenter', 'calcRfBandwidth',
        'scaleGrad', 'addGradients', 'splitGradient', 'splitGradientAt',
        'align', 'addRamps', 'pts2waveform',
        
        # Test and validation
        'testSequence', 'testGA', 'testGA1', 'testGA2',
        
        # Shape operations
        'compressShape', 'decompressShape', 'restoreAdditionalShapeSamples',
        
        # Transform and math
        'rotate3D', 'transform', 'traj2grad', 'sinc', 'gauss',
        
        # System functions
        'getSupportedRfUse', 'getSupportedLabels', 'simRf',
        
        # Constructors
        'Sequence', 'EventLibrary', 'TransformFOV', 'SeqPlot',
        
        # Complete the remaining functions
        'accurate_mod_pp', 'block2events', 'calcAdcSeg', 'calcRamp',
        'calcShortestParamsForArea', 'compressShape_mat', 'conjugate',
        'convert', 'div_check', 'extract_time', 'findFlank', 'fromRotMat',
        'generate_breaks_coefs', 'getGradAbsMag', 'init', 'InsideLimits',
        'is_grad_const', 'isOctave', 'joinleft0', 'joinleft1', 'joinright0',
        'joinright1', 'local_frac', 'local_mod', 'localStrip', 'make_matlab_doc',
        'makemosaic', 'max_abs', 'md5', 'melodyToPitchesAndDurations',
        'melodyToScale', 'multiply', 'musicToSequence', 'normalize',
        'parsemr', 'quat_conj', 'quat_multiply', 'readasc', 'rotate',
        'setup', 'sin_mix', 'test_sequence', 'testCalcADC',
        'testEventCombinations2', 'testEventCombinations3',
        'testGradientContinuity1', 'testNoEvent', 'testSingleEvents',
        'toRotMat', 'unknown', 'version'
    },
    
    'class_methods': {
        'Sequence': {
            'addBlock', 'write', 'plot', 'calculateKspacePP', 'checkTiming',
            'duration', 'read', 'setDefinition', 'getDefinition', 'getBlock',
            'setBlock', 'registerLabelEvent', 'registerRfEvent', 'registerGradEvent',
            'registerAdcEvent', 'registerControlEvent', 'registerRfShimEvent',
            'registerSoftDelayEvent', 'evalLabels', 'calcPNS', 'calcRfPower',
            'calcMomentsBtensor', 'applySoftDelay', 'flipGradAxis', 'modGradAxis',
            'asc_to_hw', 'calculateKspaceUnfunc', 'fillPpCoefs', 'findBlockByTime',
            'getBinaryCodes', 'getDefaultSoftDelayValues', 'getExtensionTypeID',
            'getExtensionTypeString', 'getRawBlockContentIDs', 'install',
            'md5_java', 'paperPlot', 'readBinary', 'rfFromLibData',
            'setExtensionStringAndID', 'sintlookup', 'slookup', 'sound',
            'testReport', 'waveforms_and_times', 'write_v141', 'writeBinary'
        },
        'EventLibrary': {'find_mat'},
        'SeqPlot': {'DataTipHandler', 'guiResize', 'updateGuides'},
        'TransformFOV': {'applyToBlock', 'applyToSeq'}
    }
}

COMMON_HALLUCINATIONS = {
    # Creation misconceptions
    'createRFPulse': 'makeSincPulse',
    'createGradient': 'makeTrapezoid',
    'createADC': 'makeAdc',
    'createDelay': 'makeDelay',
    'makeRF': 'makeSincPulse',
    'makeRfPulse': 'makeSincPulse',
    'createTrapezoid': 'makeTrapezoid',
    
    # Method misconceptions
    'addDelay': 'makeDelay',
    'setTR': 'setDefinition',
    'setTE': 'setDefinition',
    
    # Functions that don't exist in Pulseq
    'calculateSAR': None,
    'calcSNR': None,
    'makeCoil': None,
    
    # Naming variations
    'makeRectPulse': 'makeBlockPulse',
    'makeSquarePulse': 'makeBlockPulse',
    'makeGradientTrapezoid': 'makeTrapezoid',
    'makePulse': 'makeSincPulse',
    'makeReadout': 'makeAdc',
    'makeAcquisition': 'makeAdc',
    
    # Class method confusion
    'mr.write': 'seq.write',
    'mr.addBlock': 'seq.addBlock',
    'mr.plot': 'seq.plot',
    'mr.duration': 'seq.duration',
}

FUNCTION_CLUSTERS = {
    'readout_pattern': ['makeAdc', 'makeTrapezoid', 'addBlock'],
    'rf_excitation': ['makeSincPulse', 'makeTrapezoid', 'addBlock'],
    'rf_refocus': ['makeBlockPulse', 'makeDelay', 'addBlock'],
    'gradient_spoiling': ['makeTrapezoid', 'scaleGrad', 'addBlock'],
    'sequence_setup': ['Sequence', 'opts', 'setDefinition'],
    'sequence_output': ['checkTiming', 'plot', 'write'],
    'k_space': ['calculateKspacePP', 'plot'],
    'epi_readout': ['makeTrapezoid', 'makeAdc', 'scaleGrad', 'addGradients'],
    'diffusion_weighting': ['makeTrapezoid', 'makeDelay', 'calcDuration'],
    'label_control': ['makeLabel', 'registerLabelEvent', 'evalLabels'],
}