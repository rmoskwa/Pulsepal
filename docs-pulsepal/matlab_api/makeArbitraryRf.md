# makeArbitraryRf

Creates an arbitrary RF pulse with a specified shape and parameters.  The function takes a complex signal representing the desired pulse shape and calculates the corresponding RF pulse parameters, including frequency and phase offsets, and optionally generates a slice-selective gradient.

## Syntax

```matlab
function [rf, gz, gzr, delay] = makeArbitraryRf(signal,flip,varargin)
```

## Calling Pattern

```matlab
mr.makeArbitraryRf(...)
mr.makeArbitraryRf('ParameterName', value, ...)
```

## Parameters

This function accepts both positional parameters and name-value pairs.

### Required Parameters

| Parameter Name | Value Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `signal` | double | A complex-valued vector representing the desired RF pulse shape.  The amplitude of each element corresponds to the RF amplitude at a specific time point. The length of the vector determines the number of time points. | `[1+1i, 0.5+0.5i, 0, -0.5-0.5i, -1-1i]` |  |
| `flip` | double | The desired flip angle of the RF pulse in radians. | `pi/2` | radians |

### Name-Value Pair Arguments
| Parameter Name (string) | Value Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `system` | struct | `[]` | A structure containing system parameters (e.g., from mr.opts()). If empty, default system parameters are used. | `mr.opts()` |
| `freqOffset` | double | `0` | Frequency offset of the RF pulse in Hz. (Units: Hz) | `100` |
| `phaseOffset` | double | `0` | Phase offset of the RF pulse in radians. (Units: radians) | `pi/4` |
| `freqPPM` | double | `0` | Frequency offset specified in parts per million (ppm) relative to the Larmor frequency.  Can be combined with 'freqOffset'. (Units: ppm) | `10` |
| `phasePPM` | double | `0` | Phase offset specified in ppm relative to the Larmor frequency. Can be combined with 'phaseOffset'. (Units: ppm) | `5` |
| `timeBwProduct` | double | `0` | Time-bandwidth product of the pulse. (Relationship to pulse duration and bandwidth depends on pulse shape) (Units: seconds) | `4` |
| `bandwidth` | double | `0` | Bandwidth of the RF pulse in Hz. Required for slice-selective gradient calculation. (Units: Hz) | `10000` |
| `center` | double | `NaN` | Time point at which pulse is centered. If NaN, pulse is centered at its midpoint. (Units: seconds) | `0.002` |
| `maxGrad` | double | `0` | Maximum gradient amplitude in Hz/m.  Used for gradient calculation. (Units: Hz/m) | `1000` |
| `maxSlew` | double | `0` | Maximum gradient slew rate in Hz/m/s. Used for gradient calculation. (Units: Hz/m/s) | `100000` |
| `sliceThickness` | double | `0` | Thickness of the slice to be excited in meters. Used for gradient calculation. (Units: meters) | `0.005` |
| `delay` | double | `0` | Delay before the start of the RF pulse in seconds. (Units: seconds) | `0.001` |
| `dwell` | double | `0` | Time resolution (dwell time) of the RF pulse.  If 0, the system's default rfRasterTime is used. (Units: seconds) | `4e-6` |
| `use` | char | `'u'` | Specifies the purpose of the pulse ('excitation', 'refocusing', etc.). Valid values: mr.getSupportedRfUse() | `'excitation'` |

## Returns

| Output | Value Type | Description |
|--------|------|-------------|
| `rf` | struct | A structure containing the RF pulse parameters (signal, timing, frequency/phase offsets, deadtime, ringdownTime, delay, and use). |
| `gz` | struct | A structure containing the slice-selective gradient waveform (if bandwidth and sliceThickness are specified). |
| `gzr` | struct | Reserved for future use. |
| `delay` | double | Delay before the start of the RF pulse in seconds. |

## Examples

```matlab
% Create arbitrary RF pulse from predefined signal shape
rf = mr.makeArbitraryRf(ex.signal, alpha/180*pi, 'system', sys);

% Create selective RF pulse with k-space trajectory
rf = mr.makeArbitraryRf(signal, 20*pi/180, 'system', lims, 'use', 'excitation');
```

## See Also

[Sequence.makeSincPulse](makeSincPulse.md), [Sequence.addBlock](addBlock.md), [mr.opts](opts.md)
