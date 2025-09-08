# makeExtendedTrapezoid

Creates an extended trapezoid gradient waveform.  This function allows the definition of a gradient by specifying amplitude values at specific time points. It handles system limits (maxGrad, maxSlew) and ensures that the resulting waveform adheres to the gradient raster of the specified system.  The function can either return an arbitrary gradient object (if `convert2arbitrary` is true), representing the waveform on a regularly sampled grid, or it can return a gradient with potentially irregular sampling (if `convert2arbitrary` is false).

## Syntax

```matlab
function grad = makeExtendedTrapezoid(channel, varargin)
```

## Calling Pattern

```matlab
mr.makeExtendedTrapezoid(...)
mr.makeExtendedTrapezoid('ParameterName', value, ...)
```

## Parameters

This function accepts both positional parameters and name-value pairs.

### Required Parameters

| Parameter Name | Value Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `channel` | char | The gradient channel ('x', 'y', or 'z') for the waveform. | `'x'` |  |

### Name-Value Pair Arguments
| Parameter Name (string) | Value Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `'times'` | double | `0` | A vector of time points specifying the instants at which the gradient amplitudes are defined. Valid values: Must be in ascending order, and all values must be distinct. The last time point must be on a gradient raster. (Units: seconds) | `[0, 0.001, 0.002, 0.003]` |
| `'amplitudes'` | double | `0` | A vector of gradient amplitudes corresponding to the time points in 'times'. Valid values: Must have the same length as 'times'. (Units: Hz/m) | `[0, 1000, 1000, 0]` |
| `'system'` | struct | `[]` | A structure containing system parameters (e.g., from mr.opts()). If empty, default system parameters are used. | `mr.opts()` |
| `'maxGrad'` | double | `0` | Maximum gradient amplitude. If 0, the system's maxGrad is used. (Units: Hz/m) | `2000` |
| `'maxSlew'` | double | `0` | Maximum gradient slew rate. If 0, the system's maxSlew is used. (Units: Hz/m/s) | `1000000` |
| `'skip_check'` | logical | `false` | If true, skips checks for consistency between the first amplitude and the preceding block. Use with caution! | `true` |
| `'convert2arbitrary'` | logical | `false` | If true, converts the gradient to an arbitrary gradient object, resampling it onto a regular grid based on the system's gradient raster time. If false, the gradient is defined with the specified possibly irregular sampling of `times`. | `true` |

## Returns

| Output | Value Type | Description |
|--------|------|-------------|
| `grad` | struct | An arbitrary gradient object representing the extended trapezoid waveform.  The structure of this object depends on whether `convert2arbitrary` is true or false. |

## Examples

```matlab
% Example 1: Create complex gradient waveform for slice selection
GS1times = [0 GSex.riseTime];
GS1amp = [0 GSex.amplitude];
GS1 = mr.makeExtendedTrapezoid('z', 'times', GS1times, 'amplitudes', GS1amp);

% Example 2: Create readout gradient with multiple segments
GR5times = [0 GRspr.riseTime GRspr.riseTime+GRspr.flatTime GRspr.riseTime+GRspr.flatTime+GRspr.fallTime];
GR5amp = [0 GRspr.amplitude GRspr.amplitude GRacq.amplitude];
GR5 = mr.makeExtendedTrapezoid('x', 'times', GR5times, 'amplitudes', GR5amp);

% Example 3: Create composite gradient combining multiple parts
g_refC = mr.makeExtendedTrapezoid(g_ref_pre.channel, ...
    'times', [g_ref_pre.tt g_ref_post.tt+g_ref_pre.shape_dur+g_ref.flatTime], ...
    'amplitudes', [g_ref_pre.waveform g_ref_post.waveform], 'system', system);
```

## See Also

[Sequence.addBlock](addBlock.md), [mr.opts](opts.md), [makeTrapezoid](makeTrapezoid.md), [mr.pts2waveform](pts2waveform.md), [mr.makeArbitraryGrad](makeArbitraryGrad.md)
