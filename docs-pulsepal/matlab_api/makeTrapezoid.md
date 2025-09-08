# makeTrapezoid

Generates a trapezoidal gradient waveform for a specified channel in a Pulseq sequence.  The function offers flexibility in defining the waveform, allowing specification through various parameter combinations: total duration and area, flat-top duration and area, or amplitude.  It automatically calculates the necessary ramp times based on system limits (maxSlew, maxGrad) if not explicitly provided.  The function handles different scenarios and input combinations, ensuring a valid trapezoidal waveform is produced.

## Syntax

```matlab
function grad=makeTrapezoid(channel, varargin)
```

## Calling Pattern

```matlab
mr.makeTrapezoid(...)
mr.makeTrapezoid('ParameterName', value, ...)
```

## Parameters

This function accepts both positional parameters and name-value pairs.

### Required Parameters

| Parameter Name | Value Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `channel` | char | Specifies the gradient channel ('x', 'y', or 'z') for the trapezoid. | `'x'` |  |

### Name-Value Pair Arguments
| Parameter Name (string) | Value Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `system` | struct | `[]` | A structure containing system limits (e.g., maxSlew, maxGrad, gradRasterTime). If empty, defaults to mr.opts(). | `mr.opts()` |
| `'duration'` | double | `0` | Total duration of the trapezoid, including ramps.  Must be > 0 when used. (Units: seconds) | `0.005` |
| `'area'` | double | `[]` | Total area of the trapezoid, including ramps. (Units: 1/m) | `0.01` |
| `'flatTime'` | double | `[]` | Duration of the flat-top portion of the trapezoid. (Units: seconds) | `0.003` |
| `'flatArea'` | double | `[]` | Area of the flat-top portion of the trapezoid (excluding ramps). (Units: 1/m) | `0.008` |
| `'amplitude'` | double | `[]` | Amplitude of the flat-top portion of the trapezoid. (Units: Hz/m) | `1000` |
| `'maxGrad'` | double | `0` | Maximum gradient amplitude allowed. If not specified, defaults to system.maxGrad. (Units: Hz/m) | `2000` |
| `'maxSlew'` | double | `0` | Maximum gradient slew rate allowed. If not specified, defaults to system.maxSlew. (Units: Hz/m/s) | `200000` |
| `'riseTime'` | double | `0` | Time it takes for the gradient to rise to its amplitude. If not specified, it will be calculated based on maxSlew and amplitude. (Units: seconds) | `0.001` |
| `'fallTime'` | double | `0` | Time it takes for the gradient to fall from its amplitude to zero. If not specified, it will be equal to riseTime. (Units: seconds) | `0.001` |
| `'delay'` | double | `0` | Delay before the trapezoid starts. (Units: seconds) | `0.002` |

## Returns

| Output | Value Type | Description |
|--------|------|-------------|
| `grad` | struct | A Pulseq gradient waveform structure representing the trapezoid. |

## Examples

```matlab
% Readout gradient with flat area and time
gx = mr.makeTrapezoid('x', sys, 'FlatArea', Nx*deltak, 'FlatTime', roDuration);

% Pre-phasing gradient with specific area and duration
gxPre = mr.makeTrapezoid('x', sys, 'Area', -gx.area/2, 'Duration', 1e-3);

% Slice rephasing gradient
gzReph = mr.makeTrapezoid('z', lims, 'Area', -gz.area/2, 'Duration', preTime);
```

## See Also

[Sequence.addBlock](addBlock.md), [mr.opts](opts.md)
