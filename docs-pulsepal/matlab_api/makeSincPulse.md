# makeSincPulse

Generates a slice-selective sinc pulse and optionally its corresponding slice selection and refocusing gradients.  It allows for specifying various parameters to control the pulse's characteristics, including flip angle, duration, frequency and phase offsets, and gradient limits.

## Syntax

```matlab
function [rf, gz, gzr, delay] = makeSincPulse(flip,varargin)
```

## Calling Pattern

```matlab
mr.makeSincPulse(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `flip` | double | Desired flip angle of the RF pulse. | `pi/2` | radians |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `system` | struct | `[]` | System limits struct (e.g., from mr.opts()). If empty, uses default system parameters. | `mr.opts()` |
| `duration` | double | `0` | Duration of the sinc pulse. (Units: seconds) | `0.004` |
| `freqOffset` | double | `0` | Frequency offset of the RF pulse. (Units: Hz) | `100` |
| `phaseOffset` | double | `0` | Phase offset of the RF pulse. (Units: radians) | `pi/4` |
| `freqPPM` | double | `0` | Frequency offset specified in parts per million (PPM) relative to the Larmor frequency. (Units: ppm) | `-3.3` |
| `phasePPM` | double | `0` | Phase offset specified in parts per million (PPM). (Units: ppm) | `0` |
| `timeBwProduct` | double | `4` | Time-bandwidth product of the sinc pulse. | `6` |
| `apodization` | double | `0` | Apodization factor (0 for rectangular window, 1 for Hamming window). | `0.5` |
| `centerpos` | double | `0.5` | Relative position of the pulse center within its duration (0-1). | `0.7` |
| `maxGrad` | double | `0` | Maximum gradient amplitude. (Units: Hz/m) | `100e6` |
| `maxSlew` | double | `0` | Maximum gradient slew rate. (Units: Hz/m/s) | `100e6` |
| `sliceThickness` | double | `0` | Slice thickness for slice-selective excitation. (Units: meters) | `0.005` |
| `delay` | double | `0` | Additional delay after the pulse. (Units: seconds) | `0.001` |
| `dwell` | double | `0` | RF pulse dwell time (if 0, uses system.rfRasterTime). (Units: seconds) | `1e-6` |
| `use` | char | `'u'` | Specifies the pulse use ('u' for excitation, 'r' for refocusing). Valid values: ['u', 'r'] | `'excitation'` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `rf` | struct | Structure containing the RF pulse waveform. |
| `gz` | struct | Structure containing the slice-selection gradient waveform (optional). |
| `gzr` | struct | Structure containing the slice-refocusing gradient waveform (optional). |
| `delay` | double | Delay added after the pulse (seconds). |

## Examples

```matlab
% Standard excitation pulse with slice selection
[rf, gz] = mr.makeSincPulse(alpha*pi/180, 'Duration', 3e-3, ...
    'SliceThickness', sliceThickness, 'apodization', 0.42, 'timeBwProduct', 4, ...
    'use', 'excitation');

% 90-degree pulse for spin echo
[rf, gz] = mr.makeSincPulse(pi/2, 'system', lims, 'Duration', 3e-3, ...
    'SliceThickness', thickness, 'apodization', 0.5, 'timeBwProduct', 4, ...
    'use', 'excitation');

% Refocusing pulse with custom parameters
[rf180, gz180] = mr.makeSincPulse(pi, 'system', sys, 'Duration', 5e-3, ...
    'SliceThickness', sliceThickness, 'apodization', 0.5, 'timeBwProduct', 4, ...
    'use', 'refocusing');
```

## See Also

[mr.opts](opts.md), `sinc`
