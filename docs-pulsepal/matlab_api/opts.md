# opts

Sets gradient limits and other system properties of the MR system.  It allows users to define or retrieve system parameters such as maximum gradient strength, slew rate, RF pulse parameters, and timing parameters. The function uses an input parser for flexible parameter specification using name-value pairs and incorporates default values for system characteristics.

## Syntax

```matlab
function out=opts(varargin)
```

## Calling Pattern

```matlab
mr.opts(...)
mr.opts('ParameterName', value, ...)
```

## Parameters

This function accepts parameters as name-value pairs. Specify the parameter name as a string followed by its corresponding value.

### Name-Value Pair Arguments

| Parameter Name (string) | Value Type | Default | Description | Example Value |
|------|------|---------|-------------|---------|
| `'gradUnit'` | string | `'Hz/m'` | Specifies the units for maximum gradient amplitude. Valid values: `'Hz/m'`, `'mT/m'`, `'rad/ms/mm'` | `'mT/m'` |
| `'slewUnit'` | string | `'Hz/m/s'` | Specifies the units for maximum slew rate. Valid values: `'Hz/m/s'`, `'mT/m/ms'`, `'T/m/s'`, `'rad/ms/mm/ms'` | `'T/m/s'` |
| `'b1Unit'` | string | `'Hz'` | Specifies the units for maximum B1 amplitude. Valid values: `'Hz'`, `'T'`, `'mT'`, `'uT'` | `'uT'` |
| `'maxGrad'` | double | `40 mT/m` (converted to Hz/m) | Specifies the maximum gradient amplitude | `40` |
| `'maxSlew'` | double | `170 T/m/s` (converted to Hz/m/s) | Specifies the maximum gradient slew rate | `170` |
| `'maxB1'` | double | `20 uT` (converted to Hz) | Specifies the maximum RF amplitude | `20` |
| `'riseTime'` | double | `[]` | Specifies the rise time of the gradient waveforms (seconds) | `0.001` |
| `'rfDeadTime'` | double | `0` | Specifies the dead time after an RF pulse (seconds) | `0` |
| `'rfRingdownTime'` | double | `0` | Specifies the ringdown time after an RF pulse (seconds) | `0` |
| `'adcDeadTime'` | double | `0` | Specifies the dead time after ADC sampling (seconds) | `0` |
| `'adcRasterTime'` | double | `100e-9` | Specifies the raster time for ADC sampling (seconds) | `100e-9` |
| `'rfRasterTime'` | double | `1e-6` | Specifies the raster time for RF pulses (seconds) | `1e-6` |
| `'gradRasterTime'` | double | `10e-6` | Specifies the raster time for gradient waveforms (seconds) | `10e-6` |
| `'blockDurationRaster'` | double | `10e-6` | Specifies the raster time for a block of events (seconds) | `10e-6` |
| `'adcSamplesLimit'` | double | `0` | Specifies the maximum number of ADC samples (0 = no limit) | `0` |
| `'rfSamplesLimit'` | double | `0` | Specifies the maximum number of RF samples (0 = no limit) | `0` |
| `'adcSamplesDivisor'` | double | `4` | Specifies the divisor for ADC samples. The actual number of samples should be an integer multiple of this divisor | `4` |
| `'gamma'` | double | `42576000` | Specifies the gyromagnetic ratio (Hz/T) | `42576000` |
| `'B0'` | double | `1.5` | Specifies the main magnetic field strength (T) | `1.5` |
| `'setAsDefault'` | logical | `false` | If true, sets the specified parameters as the new default options | `true` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `out` | struct | A structure containing the system parameters. |

## Examples

```matlab
% Standard system limits for clinical scanners
sys = mr.opts('MaxGrad', 28, 'GradUnit', 'mT/m', 'MaxSlew', 150, 'SlewUnit', 'T/m/s', 'riseTime', 0.0001);

% High-performance gradients for research systems
sys = mr.opts('MaxGrad', 32, 'GradUnit', 'mT/m', 'MaxSlew', 130, 'SlewUnit', 'T/m/s', 'riseTime', 250e-6);

% System limits for spectroscopy sequences
system = mr.opts('MaxGrad', 15, 'GradUnit', 'mT/m', 'MaxSlew', 100, 'SlewUnit', 'T/m/s', 'riseTime', 0.0001);
```

## See Also

[mr.convert](convert.md)
