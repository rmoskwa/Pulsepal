# makeGaussPulse

Generates a Gaussian RF pulse, optionally with slice-selective gradients.  The function creates a Gaussian RF pulse with specified flip angle and duration. It allows for adjustments in frequency and phase offsets (in Hz and radians, or as a percentage of the Larmor frequency in ppm), and includes options for apodization and precise control over bandwidth and time-bandwidth product.  Additionally, it can generate slice-selective gradients (gz) and corresponding refocusing gradients (gzr), given slice thickness and gradient limits (maxGrad, maxSlew).

## Syntax

```matlab
function [rf, gz, gzr, delay] = makeGaussPulse(flip,varargin)
```

## Calling Pattern

```matlab
mr.makeGaussPulse(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `flip` | double | Desired flip angle of the RF pulse. | `pi/2` | radians |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `system` | struct | `[]` | Pulseq system structure containing hardware parameters (e.g., from mr.opts()). If empty, uses default system parameters. | `mr.opts()` |
| `duration` | double | `0` | Duration of the RF pulse. (Units: seconds) | `0.004` |
| `freqOffset` | double | `0` | Frequency offset of the RF pulse in Hz. (Units: Hz) | `100` |
| `phaseOffset` | double | `0` | Phase offset of the RF pulse in radians. (Units: radians) | `pi/4` |
| `freqPPM` | double | `0` | Frequency offset specified in parts per million (ppm) relative to the Larmor frequency.  Useful for fat saturation. (Units: ppm) | `-3.3` |
| `phasePPM` | double | `0` | Phase offset specified in parts per million (ppm) relative to the Larmor frequency. (Units: ppm) | `0` |
| `timeBwProduct` | double | `3` | Time-bandwidth product of the Gaussian pulse. | `4` |
| `bandwidth` | double | `0` | Bandwidth of the RF pulse in Hz (overrides timeBwProduct if specified). (Units: Hz) | `1000` |
| `apodization` | double | `0` | Apodization parameter (0 for no apodization, 1 for full Hamming window). | `0.5` |
| `centerpos` | double | `0.5` | Position of the pulse center (0 to 1, where 0 is the beginning and 1 is the end). | `0.7` |
| `maxGrad` | double | `0` | Maximum gradient amplitude for slice selection. (Units: Hz/m) | `300` |
| `maxSlew` | double | `0` | Maximum gradient slew rate for slice selection. (Units: Hz/m/s) | `100000` |
| `sliceThickness` | double | `0` | Thickness of the slice for slice selection. (Units: meters) | `0.005` |
| `delay` | double | `0` | Delay before the pulse. (Units: seconds) | `0.001` |
| `dwell` | double | `0` | RF pulse dwell time. If 0, it uses the system's default rfRasterTime. (Units: seconds) | `0.000002` |
| `use` | char | `'u'` | Specifies the intended use of the RF pulse, for k-space calculation. Valid values: mr.getSupportedRfUse() | `'excitation'` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `rf` | struct | Pulseq RF pulse definition. |
| `gz` | struct | Pulseq slice selection gradient definition. |
| `gzr` | struct | Pulseq slice refocusing gradient definition. |
| `delay` | double | calculated delay |

## Examples

```matlab
[rf, gz] = mr.makeGaussPulse(pi/2, 'duration', 0.004, 'sliceThickness', 0.005, 'maxGrad', 300, 'maxSlew', 100000);
```

## See Also

`gauss`, [mr.opts](opts.md), [Sequence.addBlock](addBlock.md)
