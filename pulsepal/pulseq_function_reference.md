# Pulseq Function Reference

## Verified Pulseq Functions (89 total)
Accurate as of Pulseq v1.5.0 - MATLAB implementation

### RF Pulse Creation
- `mr.makeBlockPulse`: Generates a block RF pulse with optional slice-selective capabilities
- `mr.makeSincPulse`: Generates a slice-selective sinc pulse and optionally slice selection gradients
- `mr.makeGaussPulse`: Generates a Gaussian RF pulse, optionally with slice-selective gradients
- `mr.makeSLRpulse`: Generates selective excitation RF pulse using the Shinnar-Le Roux (SLR) algorithm
- `mr.makeAdiabaticPulse`: Generates adiabatic inversion pulses (hyperbolic secant or WURST types)
- `mr.makeArbitraryRf`: Creates an arbitrary RF pulse with a specified shape and parameters

### Gradient Waveform Creation
- `mr.makeTrapezoid`: Generates a trapezoidal gradient waveform for a specified channel
- `mr.makeExtendedTrapezoid`: Creates an extended trapezoid gradient waveform
- `mr.makeExtendedTrapezoidArea`: Generates shortest possible extended trapezoid with specified area
- `mr.makeArbitraryGrad`: Creates a gradient event with an arbitrary waveform

### Sequence Object Management
- `mr.Sequence()`: Constructor for the Sequence object in the Pulseq toolbox
- `seq.addBlock`: Adds a new block of events to a Pulseq sequence
- `seq.setBlock`: Adds or replaces a block of events within a Pulseq sequence
- `seq.getBlock`: Retrieves a specific block from a Pulseq sequence object
- `seq.duration`: Calculates the total duration of a Pulseq sequence
- `seq.getDefinition`: Retrieves user-defined parameters from the sequence definitions
- `seq.setDefinition`: Sets user-defined parameters in the sequence definitions

### Sequence File I/O
- `seq.write`: Writes a Pulseq sequence object to a file in the Pulseq open file format
- `seq.write_v141`: Writes sequence in v1.4.1 format for backward compatibility
- `seq.read`: Reads a Pulseq sequence from a .seq file
- `seq.writeBinary`: Writes sequence in binary format for faster I/O
- `seq.readBinary`: Reads sequence from binary format
- `parsemr`: Parses and displays .seq file contents as standalone function

### Data Acquisition (ADC)
- `mr.makeAdc`: Creates an ADC readout event for Pulseq sequence design
- `mr.calcAdcSeg`: Calculates ADC segmentation for non-Cartesian trajectories

### Timing Control
- `mr.makeDelay`: Creates a delay event for use in a Pulseq sequence
- `mr.makeSoftDelay`: Creates an adjustable delay for sequence optimization
- `mr.calcDuration`: Calculates the duration of Pulseq events or block structures
- `mr.align`: Aligns objects within a Pulseq block based on specified alignment
- `seq.applySoftDelay`: Applies soft delays to optimize sequence timing
- `seq.findBlockByTime`: Finds block index at specific time point in sequence

### Gradient Manipulation
- `mr.scaleGrad`: Scales a gradient waveform by a given scalar value
- `mr.addGradients`: Superposes multiple gradient waveforms
- `mr.splitGradient`: Splits a gradient at specified point into two parts
- `mr.splitGradientAt`: Splits gradient at specific time point
- `mr.addRamps`: Adds ramp-up/down to existing gradient waveform
- `mr.calcRamp`: Calculates ramp time for given gradient parameters

### 3D Spatial Transformations
- `mr.rotate`: Rotates a 3D vector by a given unit quaternion for efficient spatial transformations
- `mr.rotate3D`: Rotates gradient objects according to a 3x3 rotation matrix
- `mr.transform`: Applies general transformation matrix to gradients
- `seq.modGradAxis`: Inverts or scales all gradients along specified axes
- `seq.flipGradAxis`: Flips/inverts gradient polarity on specified axis

### Trajectory Conversions
- `mr.traj2grad`: Converts k-space trajectory to gradient waveforms
- `seq.calculateKspacePP`: Calculates k-space trajectory using piecewise-polynomial approximation

### RF Pulse Analysis
- `mr.calcRfCenter`: Calculates the effective center time point of an RF pulse
- `mr.calcRfBandwidth`: Calculates the bandwidth of an RF pulse
- `mr.simRf`: Simulates RF pulse response and slice profile
- `seq.calcRfPower`: Calculates RF power and SAR from the pulse amplitudes

### System Configuration
- `mr.opts`: Sets gradient limits and system properties for MR sequences
- `mr.getSupportedRfUse`: Returns list of supported RF pulse use types
- `mr.getSupportedLabels`: Returns list of supported sequence labels
- `mr.convert`: Converts between different units
- `mr.version`: Returns Pulseq version information

### Sequence Validation & Analysis
- `seq.checkTiming`: Checks timing of all blocks and objects within the sequence
- `seq.testReport`: Generates comprehensive test report for sequence validation
- `seq.calcPNS`: Calculates PNS levels for the sequence
- `seq.calcMomentsBtensor`: Calculates gradient moments and b-tensor for diffusion

### Visualization & Output
- `seq.plot`: Plots a Pulseq sequence object in a new figure
- `seq.paperPlot`: Generates publication-quality sequence diagram
- `seq.sound`: Generates audio representation of sequence for debugging
- `seq.waveforms_and_times`: Extracts waveforms and timing arrays for custom plotting

### Hardware Control
- `mr.makeLabel`: Creates a label event for hardware synchronization and loop control
- `mr.makeTrigger`: Creates a trigger event for external device synchronization
- `mr.makeDigitalOutputPulse`: Creates a digital output pulse for hardware control
- `seq.evalLabels`: Evaluates label values throughout the sequence

### Waveform Processing
- `mr.pts2waveform`: Converts discrete time points to continuous waveform
- `mr.compressShape`: Compresses waveform data for efficient storage
- `mr.compressShape_mat`: MATLAB-specific waveform compression algorithm
- `mr.decompressShape`: Decompresses previously compressed waveform data
- `mr.restoreAdditionalShapeSamples`: Restores samples removed during compression

### Event Registration (Advanced)
- `seq.registerGradEvent`: Registers gradient event in internal library for reuse
- `seq.registerRfEvent`: Registers RF event in internal library for reuse
- `seq.registerLabelEvent`: Registers label event in internal library for reuse

### Utility Classes
- `mr.EventLibrary()`: Constructor for event library object for internal storage
- `mr.TransformFOV()`: Creates FOV transformation object for spatial transformations
- `mr.SeqPlot()`: Creates sequence plotting object for GUI display

### Quaternion Operations (mr.aux.quat namespace)
- `mr.aux.quat.q2rot`: Convert quaternion to rotation matrix
- `mr.aux.quat.rot2q`: Convert rotation matrix to quaternion
- `mr.aux.quat.qconj`: Calculate quaternion conjugate
- `mr.aux.quat.qmult`: Multiply two quaternions
- `mr.aux.quat.normalize`: Normalize quaternion to unit length

### Additional Utilities
- `mr.aux.findFlank`: Find rising/falling edge in waveform
- `mr.aux.isOctave`: Check if running in Octave
- `mr.Siemens.readasc`: Read Siemens ASC parameter files
- `mrMusic.musicToSequence`: Convert musical notes to MRI sequence
