# makeExtendedTrapezoidArea

Generates the shortest possible extended trapezoid gradient waveform with a specified area, starting and ending with optionally non-zero gradient values.  The function uses optimization techniques (fminsearch) to find the optimal gradient amplitude and plateau duration to achieve the desired area while respecting system limits (maximum gradient amplitude and slew rate).

## Syntax

```matlab
function [grad, times, amplitudes] = makeExtendedTrapezoidArea(channel, Gs, Ge, A, sys)
```

## Calling Pattern

```matlab
mr.makeExtendedTrapezoidArea(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `channel` | char | The gradient channel ('x', 'y', or 'z') for which to create the waveform. | `'x'` |  |
| `Gs` | double | The starting gradient amplitude. | `100` | Hz/m |
| `Ge` | double | The ending gradient amplitude. | `100` | Hz/m |
| `A` | double | The desired area of the extended trapezoid. | `0.1` | 1/m |
| `sys` | struct | A structure containing system parameters.  Must include fields like 'maxSlew' (maximum slew rate in Hz/m/s), 'gradRasterTime' (gradient raster time in seconds), and 'maxGrad' (maximum gradient amplitude in Hz/m). | `mr.opts()` |  |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `grad` | struct | A Pulseq gradient waveform structure containing the generated extended trapezoid. |
| `times` | double | A vector of time points (in seconds) defining the waveform. |
| `amplitudes` | double | A vector of gradient amplitudes (in Hz/m) corresponding to the time points. |

## Examples

```matlab
[grad, times, amplitudes] = mr.makeExtendedTrapezoidArea('x', 100, 100, 0.1, mr.opts())
```

## See Also

[mr.makeExtendedTrapezoid](makeExtendedTrapezoid.md)
