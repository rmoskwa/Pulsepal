# makeSLRpulse

Generates a selective excitation RF pulse using the Shinnar-Le Roux (SLR) algorithm. This function acts as a wrapper for a Python function that utilizes the sigpy library. It designs RF pulses for various applications, including excitation, refocusing, and inversion.

## Syntax

```matlab
function [rf, gz, gzr, delay] = makeSLRpulse(flip,varargin)
```

## Calling Pattern

```matlab
mr.makeSLRpulse(...)
mr.makeSLRpulse('ParameterName', value, ...)
```

## Parameters

This function accepts both positional parameters and name-value pairs.

### Required Parameters

| Parameter Name | Value Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `flip` | double | The desired flip angle of the RF pulse. | `pi/2` | radians |

### Name-Value Pair Arguments
| Parameter Name (string) | Value Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `system` | struct | `[]` | A structure containing system parameters (e.g., from mr.opts()).  If empty, default parameters will be used. | `mr.opts()` |
| `duration` | double | `1e-3` | The duration of the RF pulse. (Units: seconds) | `0.004` |
| `freqOffset` | double | `0` | Frequency offset of the RF pulse. (Units: Hz) | `100` |
| `phaseOffset` | double | `0` | Phase offset of the RF pulse. (Units: radians) | `pi/4` |
| `freqPPM` | double | `0` | Frequency offset in parts per million (ppm). (Units: ppm) | `10` |
| `phasePPM` | double | `0` | Phase offset in parts per million (ppm). (Units: ppm) | `5` |
| `timeBwProduct` | double | `4` | Time-bandwidth product of the pulse. | `6` |
| `passbandRipple` | double | `0.01` | Passband ripple of the filter. | `0.005` |
| `stopbandRipple` | double | `0.01` | Stopband ripple of the filter. | `0.005` |
| `filterType` | char | `'mt'` | Type of filter used for pulse design ('mt', 'ms', 'pm', 'min', 'max', 'ls'). | `'ls'` |
| `apodization` | double | `0` | Apodization parameter (currently not used). | `0.1` |
| `centerpos` | double | `0.5` | Center position (currently not used). | `0.6` |
| `maxGrad` | double | `0` | Maximum gradient amplitude. (Units: Hz/m) | `2000` |
| `maxSlew` | double | `0` | Maximum gradient slew rate. (Units: Hz/m/s) | `100000` |
| `sliceThickness` | double | `0` | Slice thickness. (Units: meters) | `0.005` |
| `delay` | double | `0` | Additional delay before the pulse. (Units: seconds) | `0.001` |
| `dwell` | double | `0` | Dwell time (currently not used). (Units: seconds) | `0.000001` |
| `use` | char | `'excitation'` | Pulse type ('excitation', 'refocusing', etc.). | `'refocusing'` |
| `pythonCmd` | char | `''` | Python command (currently not used). | `''` |

## Returns

| Output | Value Type | Description |
|--------|------|-------------|
| `rf` | struct | The designed RF pulse (Pulseq sequence). |
| `gz` | struct | The slice-selective gradient (Pulseq sequence). |
| `gzr` | struct | The refocusing gradient (Pulseq sequence, might be empty). |
| `delay` | double | The total delay associated with the pulse |

## Examples

```matlab
% 90-degree excitation pulse with detailed control
[rf_ex, gz, gzr] = mr.makeSLRpulse(pi/2, 'duration', rfDur1, 'SliceThickness', sliceThickness*sth_ex, ...
    'timeBwProduct', 5, 'dwell', rfDur1/500, 'passbandRipple', 1, 'stopbandRipple', 1e-2, ...
    'filterType', 'ms', 'system', system, 'use', 'excitation', 'PhaseOffset', pi/2);

% 180-degree refocusing pulse
[rf_ref, g_ref] = mr.makeSLRpulse(pi, 'duration', rfDurRef, 'PhaseOffset', pi/2, ...
    'SliceThickness', voxel(2), 'timeBwProduct', 6, 'passbandRipple', 1, 'stopbandRipple', 1e-2, ...
    'filterType', 'ms', 'system', system, 'use', 'refocusing');

% Multi-slice excitation
[rf_ex, g_ex, g_exReph] = mr.makeSLRpulse(pi/2, 'Duration', rfDurEx, ...
    'SliceThickness', voxel(1), 'timeBwProduct', 6, 'passbandRipple', 1, 'stopbandRipple', 1e-2, ...
    'filterType', 'ms', 'system', system, 'use', 'excitation');
```

## See Also

[mr.opts](opts.md)
