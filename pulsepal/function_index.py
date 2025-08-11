"""
Hardcoded index of all 150 MATLAB Pulseq functions.
Generated from function_calling_patterns table in Supabase.
"""

MATLAB_FUNCTIONS = {
    "direct_calls": {
        # Core creation functions (most commonly used)
        "makeAdc",
        "makeTrapezoid",
        "makeDelay",
        "makeSincPulse",
        "makeGaussPulse",
        "makeBlockPulse",
        "makeArbitraryGrad",
        "makeArbitraryRf",
        "makeExtendedTrapezoid",
        "makeExtendedTrapezoidArea",
        "makeSLRpulse",
        "makeAdiabaticPulse",
        "makeLabel",
        "makeTrigger",
        "makeDigitalOutputPulse",
        "makeSoftDelay",
        # Utility functions
        "opts",
        "calcDuration",
        "calcRfCenter",
        "calcRfBandwidth",
        "scaleGrad",
        "addGradients",
        "splitGradient",
        "splitGradientAt",
        "align",
        "addRamps",
        "pts2waveform",
        # Test and validation
        "testCalcADC",
        "testReport",  # Useful public method
        # Shape operations
        "compressShape",
        "decompressShape",
        "restoreAdditionalShapeSamples",
        # Transform and math
        "rotate3D",
        "transform",
        "traj2grad",
        "sinc",
        "gauss",
        # System functions
        "getSupportedRfUse",
        "getSupportedLabels",
        "simRf",
        # Constructors
        "Sequence",
        "EventLibrary",
        "TransformFOV",
        "SeqPlot",
        # Remaining public functions
        "calcAdcSeg",
        "calcRamp",
        "compressShape_mat",  # MATLAB-compatible version
        "convert",
        # Music demo functions (useful for music sequences)
        "init",  # mrMusic.init
        "melodyToPitchesAndDurations",  # mrMusic function
        "melodyToScale",  # mrMusic function
        "musicToSequence",  # mrMusic function
        # Utilities
        "md5",  # Public MD5 hash function
        "parsemr",  # Public utility for loading/displaying sequences
        "readasc",  # Siemens ASC file reader
        "version",
    },
    "class_methods": {
        "Sequence": {
            "addBlock",
            "write",
            "plot",
            "calculateKspacePP",
            "checkTiming",
            "duration",
            "read",
            "setDefinition",
            "getDefinition",
            "getBlock",
            "setBlock",
            "evalLabels",
            "calcPNS",
            "calcRfPower",
            "calcMomentsBtensor",
            "applySoftDelay",
            "flipGradAxis",
            "modGradAxis",
            "findBlockByTime",
            "install",
            "paperPlot",
            "readBinary",
            "sound",
            "testReport",
            "waveforms_and_times",
            "write_v141",
            "writeBinary",
        },
        "EventLibrary": {},  # No public methods users should call
        "SeqPlot": {},  # GUI internals - users don't call these directly
        "TransformFOV": {"applyToBlock", "applyToSeq"},
    },
    # Additional prefixed functions
    "eve_functions": {},  # No public EventLibrary methods
    "tra_functions": {"applyToBlock", "applyToSeq"},  # tra.* functions (TransformFOV)
    "mr_aux_functions": {
        "findFlank",
        "isOctave",
        "version",
    },  # mr.aux.* functions
    "mr_aux_quat_functions": {  # mr.aux.quat.* functions
        "conjugate",
        "fromRotMat",
        "multiply",
        "normalize",
        "rotate",  # Quaternion rotation
        "toRotMat",
    },
}

COMMON_HALLUCINATIONS = {
    # Creation misconceptions
    "createRFPulse": "makeSincPulse",
    "createGradient": "makeTrapezoid",
    "createADC": "makeAdc",
    "createDelay": "makeDelay",
    "makeRF": "makeSincPulse",
    "makeRfPulse": "makeSincPulse",
    "createTrapezoid": "makeTrapezoid",
    # Method misconceptions
    "addDelay": "makeDelay",
    "setTR": "setDefinition",
    "setTE": "setDefinition",
    # Functions that don't exist in Pulseq
    "calculateSAR": None,
    "calcSNR": None,
    "makeCoil": None,
    # Properties that don't exist
    "gamma": None,  # mr.gamma doesn't exist
    "gammaHz": None,  # mr.gammaHz doesn't exist
    # Naming variations
    "makeRectPulse": "makeBlockPulse",
    "makeSquarePulse": "makeBlockPulse",
    "makeGradientTrapezoid": "makeTrapezoid",
    "makePulse": "makeSincPulse",
    "makeReadout": "makeAdc",
    "makeAcquisition": "makeAdc",
    # Class method confusion
    "mr.write": "seq.write",
    "mr.addBlock": "seq.addBlock",
    "mr.plot": "seq.plot",
    "mr.duration": "seq.duration",
    # K-space calculation hallucinations
    "calcKspace": "calculateKspacePP",
    "calculateKspace": "calculateKspacePP",
    "calcKSpace": "calculateKspacePP",
    "calculateKSpace": "calculateKspacePP",
    # Adiabatic pulse hallucinations
    "makeAdiaRF": "makeAdiabaticPulse",
    "makeAdiabaticRF": "makeAdiabaticPulse",
    "makeAdiabatic": "makeAdiabaticPulse",
}

FUNCTION_CLUSTERS = {
    "readout_pattern": ["makeAdc", "makeTrapezoid", "addBlock"],
    "rf_excitation": ["makeSincPulse", "makeTrapezoid", "addBlock"],
    "rf_refocus": ["makeBlockPulse", "makeDelay", "addBlock"],
    "gradient_spoiling": ["makeTrapezoid", "scaleGrad", "addBlock"],
    "sequence_setup": ["Sequence", "opts", "setDefinition"],
    "sequence_output": ["checkTiming", "plot", "write"],
    "k_space": ["calculateKspacePP", "plot"],
    "epi_readout": ["makeTrapezoid", "makeAdc", "scaleGrad", "addGradients"],
    "diffusion_weighting": ["makeTrapezoid", "makeDelay", "calcDuration"],
    "label_control": ["makeLabel", "registerLabelEvent", "evalLabels"],
}

# Comprehensive namespace mapping for validation
NAMESPACE_MAP = {
    # Functions that MUST use mr. namespace
    "mr_only": {
        # All make* functions
        "makeAdc", "makeTrapezoid", "makeDelay", "makeSincPulse", "makeGaussPulse",
        "makeBlockPulse", "makeArbitraryGrad", "makeArbitraryRf", "makeExtendedTrapezoid",
        "makeExtendedTrapezoidArea", "makeSLRpulse", "makeAdiabaticPulse", "makeLabel",
        "makeTrigger", "makeDigitalOutputPulse", "makeSoftDelay",
        # Calc functions
        "calcDuration", "calcRfCenter", "calcRfBandwidth", "calcRamp", "calcAdcSeg",
        # Other mr functions
        "opts", "scaleGrad", "addGradients", "splitGradient", "splitGradientAt",
        "align", "addRamps", "pts2waveform", "compressShape",
        "decompressShape", "rotate3D", "transform", "traj2grad", "sinc", "gauss",
        "getSupportedRfUse", "getSupportedLabels", "simRf",
    },

    # Functions that MUST use seq. namespace (Sequence methods)
    "seq_only": {
        "addBlock", "write", "plot", "calculateKspacePP", "checkTiming", "duration",
        "read", "setDefinition", "getDefinition", "getBlock", "setBlock",
        "evalLabels", "calcPNS", "calcRfPower", "calcMomentsBtensor", "applySoftDelay",
        "flipGradAxis", "modGradAxis", "testReport", "sound", "paperPlot",
        "waveforms_and_times", "writeBinary", "readBinary",
    },

    # Constructors (can be called without namespace)
    "constructors": {
        "Sequence", "EventLibrary", "TransformFOV", "SeqPlot",
    },

    # EventLibrary methods (eve. namespace)
    "eve_only": {
        "find_mat",
    },

    # TransformFOV methods (tra. namespace)
    "tra_only": {
        "applyToBlock", "applyToSeq",
    },

    # mr.aux functions
    "mr_aux_only": {
        "findFlank", "isOctave", "version",
    },

    # mr.aux.quat functions
    "mr_aux_quat_only": {
        "conjugate", "fromRotMat", "multiply", "normalize", "rotate", "toRotMat",
    },
}

def get_correct_namespace(function_name: str) -> str:
    """
    Get the correct namespace for a function.
    Returns the namespace prefix (e.g., 'mr', 'seq') or empty string if no namespace needed.
    """
    # Check each namespace category
    if function_name in NAMESPACE_MAP["mr_only"]:
        return "mr"
    if function_name in NAMESPACE_MAP["seq_only"]:
        return "seq"
    if function_name in NAMESPACE_MAP["constructors"]:
        return ""  # No namespace needed
    if function_name in NAMESPACE_MAP["eve_only"]:
        return "eve"
    if function_name in NAMESPACE_MAP["tra_only"]:
        return "tra"
    if function_name in NAMESPACE_MAP["mr_aux_only"]:
        return "mr.aux"
    if function_name in NAMESPACE_MAP["mr_aux_quat_only"]:
        return "mr.aux.quat"
    return None  # Function not found

def validate_namespace(function_name: str, provided_namespace: str = None) -> dict:
    """
    Validate if a function is being called with the correct namespace.
    
    Args:
        function_name: The function name (e.g., 'makeAdc')
        provided_namespace: The namespace provided by user (e.g., 'seq', 'mr', or None)
    
    Returns:
        dict with:
        - is_valid: Whether the namespace is correct
        - correct_form: The correct way to call this function
        - error: Error message if invalid
    """
    correct_namespace = get_correct_namespace(function_name)

    if correct_namespace is None:
        # Function doesn't exist
        return {
            "is_valid": False,
            "correct_form": None,
            "error": f"Function '{function_name}' not found in Pulseq",
        }

    # Build the correct form
    if correct_namespace:
        correct_form = f"{correct_namespace}.{function_name}"
    else:
        correct_form = function_name

    # If no namespace was provided, return the correct form
    if provided_namespace is None:
        return {
            "is_valid": True,
            "correct_form": correct_form,
            "error": None,
        }

    # Check if provided namespace matches
    if provided_namespace == correct_namespace:
        return {
            "is_valid": True,
            "correct_form": correct_form,
            "error": None,
        }
    # Wrong namespace
    provided_form = f"{provided_namespace}.{function_name}" if provided_namespace else function_name
    return {
        "is_valid": False,
        "correct_form": correct_form,
        "error": f"'{provided_form}' should be '{correct_form}'",
    }
