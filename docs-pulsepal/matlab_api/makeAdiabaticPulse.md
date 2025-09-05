# makeAdiabaticPulse

Generates adiabatic inversion pulses of type 'hypsec' (hyperbolic secant) or 'wurst' (wideband, uniform rate, smooth truncation).  It acts as a wrapper for a Python function (requiring the 'sigpy' library), creating RF and gradient waveforms. Note that this function is likely to only work on Linux systems.

## Syntax

```matlab
function [rf, gz, gzr, delay] = makeAdiabaticPulse(type,varargin)
```

## Calling Pattern

```matlab
mr.makeAdiabaticPulse(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `type` | string | Specifies the type of adiabatic pulse to generate.  Must be either 'hypsec' or 'wurst'. | `'hypsec'` |  |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `system` | struct | `[]` | System parameters (e.g., from mr.opts()).  Provides system limits for gradient waveforms (if maxGrad and maxSlew are not specified). | `mr.opts()` |
| `duration` | double | `10e-3` | Total duration of the pulse. (Units: seconds) | `0.01` |
| `freqOffset` | double | `0` | Frequency offset of the pulse. (Units: Hz) | `100` |
| `phaseOffset` | double | `0` | Phase offset of the pulse. (Units: radians) | `pi/2` |
| `freqPPM` | double | `0` | Frequency offset in parts per million. (Units: ppm) | `0.1` |
| `phasePPM` | double | `0` | Phase offset in parts per million. (Units: ppm) | `0.1` |
| `beta` | double | `800` | AM waveform parameter (relevant for 'hypsec' pulse). | `1000` |
| `mu` | double | `4.9` | Constant determining the amplitude of the frequency sweep (relevant for 'hypsec' pulse). | `5.0` |
| `n_fac` | double | `40` | Power to exponentiate within the AM term (relevant for 'wurst' pulse). | `30` |
| `bandwidth` | double | `40000` | Pulse bandwidth (relevant for 'wurst' pulse). (Units: Hz) | `20000` |
| `adiabaticity` | double | `4` | Adiabaticity factor. | `5` |
| `maxGrad` | double | `0` | Maximum gradient amplitude. (Units: Hz/m) | `1000` |
| `maxSlew` | double | `0` | Maximum gradient slew rate. (Units: Hz/m/s) | `100000` |
| `sliceThickness` | double | `0` | Slice thickness. (Units: meters) | `0.005` |
| `delay` | double | `0` | Delay after the pulse. (Units: seconds) | `0.001` |
| `dwell` | double | `0` | Dwell time. (Units: seconds) | `1e-6` |
| `use` | string | `'u'` | Specifies how the pulse will be used.  See mr.getSupportedRfUse() for options. | `'excitation'` |
| `pythonCmd` | string | `''` | Command to execute the Python function (for advanced use cases). | `''` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `rf` | struct | Pulseq RF waveform structure. |
| `gz` | struct | Pulseq slice-select gradient waveform structure. |
| `gzr` | struct | Pulseq refocusing gradient waveform structure (if applicable). |
| `delay` | double | Delay time (seconds). |

## Examples

```matlab
[rf, gz, gzr, delay] = mr.makeAdiabaticPulse('hypsec', 'duration', 0.01, 'beta', 1000, 'mu', 5);
[rf, gz, gzr, delay] = mr.makeAdiabaticPulse('wurst', 'duration', 0.005, 'bandwidth', 20000, 'n_fac', 30);
```

## See Also

[mr.opts](opts.md), [mr.makeExtendedTrapezoid](makeExtendedTrapezoid.md), [mr.getSupportedRfUse](getSupportedRfUse.md)
