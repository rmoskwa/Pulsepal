# addGradients

This function superposes multiple gradient waveforms. It takes a cell array of gradient waveforms as input and returns a single gradient waveform representing their sum.  The function handles different gradient types (trapezoids and arbitrary waveforms), checks for consistency in channel and timing, and applies system limits (maxGrad and maxSlew). If all input gradients are trapezoids with identical timing, the function efficiently sums their amplitudes.

## Syntax

```matlab
function grad = addGradients(grads, varargin)
```

## Calling Pattern

```matlab
mr.addGradients(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `grads` | cell | A cell array containing multiple gradient waveforms. Each element of the cell array should be a structure representing a single gradient waveform (e.g., as created by `makeTrapezoid`).  These structures must have at least 'channel', 'delay', and 'type' fields. | `{g1, g2, g3}` |  |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `system` | struct | `[]` | A structure containing system parameters, typically obtained from `mr.opts()`.  If not provided, `mr.opts()` is used to get default system parameters.  Relevant fields include `maxSlew` and `maxGrad`. | `mr.opts()` |
| `maxGrad` | double | `0` | Maximum gradient amplitude. If set to a value greater than 0, it overrides the `maxGrad` value from the `system` parameter. (Units: Hz/m) | `400` |
| `maxSlew` | double | `0` | Maximum gradient slew rate. If set to a value greater than 0, it overrides the `maxSlew` value from the `system` parameter. (Units: Hz/m/s) | `20000` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `grad` | struct | A structure representing the resulting superimposed gradient waveform. |

## Examples

```matlab
% Example 1: Combining gradient parts with prewinder
gx_parts = mr.splitGradientAt(gx, ceil((adc.dwell*adc.numSamples+adc.delay+adc.deadTime)/sys.gradRasterTime)*sys.gradRasterTime);
gx_parts(1).delay = mr.calcDuration(gxPre);
gx_1 = mr.addGradients({gxPre, gx_parts(1)}, 'system', sys);

% Example 2: Combining rephasing gradient with slice selection
gz_parts = mr.splitGradientAt(gz, rf.delay+rf.t(end));
gz_parts(1).delay = mr.calcDuration(gzReph);
gz_1 = mr.addGradients({gzReph, gz_parts(1)}, 'system', sys);

% Example 3: Adding spoiler gradients with readout components
gro1 = mr.addGradients({gro1, groPre, groSp}, 'system', sys);
gpe1c = mr.addGradients({gpe1, mr.makeTrapezoid(ax.d2, 'Area', -gpe1.area, 'duration', groSp.shape_dur, 'delay', groSp.delay, 'system', sys)});
```

## See Also

[Sequence.addBlock](addBlock.md), [mr.opts](opts.md), [makeTrapezoid](makeTrapezoid.md)
