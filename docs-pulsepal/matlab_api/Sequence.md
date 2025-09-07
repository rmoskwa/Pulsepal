# Sequence

This function is a constructor for the Sequence object in the Pulseq toolbox. It initializes a sequence object with default parameters and libraries for storing different types of events (RF pulses, gradients, ADC events, etc.).  It also sets up default raster times and provides methods for reading and writing sequence files in different formats (v1.4.1 and binary).  Additional methods are included for calculating pulse-related parameters such as PNS (peak-to-null ratio) and gradient moments.

## Syntax

```matlab
function obj = Sequence(varargin)
```

## Calling Pattern

```matlab
seq = mr.Sequence(...)
```

## Parameters


### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `trajectory_delay` | double | `0` | This parameter likely controls a delay in the sequence trajectory.  It represents the time offset applied before the beginning of the trajectory. (Units: seconds) | `0.001` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `obj` | struct | A Sequence object containing all the initialized parameters, libraries, and methods for manipulating and storing the MRI sequence information. |

## Examples

```matlab
% Create sequence object with system limits for clinical scanner
seq = mr.Sequence(sys);

% Create sequence object with default parameters
seq = mr.Sequence();

% Create sequence object for SAR testing with separate system limits
seq_sar = mr.Sequence(sys);
```

## See Also

[mr.opts](opts.md), [Sequence.read](read.md), [Sequence.write](write.md), [Sequence.write_v141](write_v141.md), [Sequence.readBinary](readBinary.md), [Sequence.writeBinary](writeBinary.md), [Sequence.calcPNS](calcPNS.md), [Sequence.calcMomentsBtensor](calcMomentsBtensor.md), [Sequence.testReport](testReport.md)
