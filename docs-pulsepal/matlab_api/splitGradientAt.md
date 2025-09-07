# splitGradientAt

Splits a trapezoidal or arbitrary gradient into two parts at a specified time point.  The function divides a gradient waveform, either a trapezoid or an arbitrary shaped gradient, into two separate gradients at a given time.  It adjusts the delays to ensure that combining the resulting gradients using `addGradients` recreates the original gradient. For trapezoidal gradients, it returns extended trapezoids; for arbitrary gradients, it returns arbitrary gradient objects.

## Syntax

```matlab
function [varargout] = splitGradientAt(grad, timepoint, varargin)
```

## Calling Pattern

```matlab
mr.splitGradientAt(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `grad` | struct | Structure representing the gradient waveform to be split.  This structure contains fields like 'type' ('grad' or 'trap'), 'tt' (time points), 'waveform' (amplitude values), 'channel', and 'delay'. | `{type:'trap', tt:[0 1 2 3], waveform:[0 100 100 0], channel:'x', delay:0}` |  |
| `timepoint` | double | Time point (in seconds) at which to split the gradient waveform. | `0.002` | seconds |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `system` | struct | `[]` | Structure containing system parameters, such as `gradRasterTime`. If not provided, default system parameters are used (via `mr.opts()`). Valid values: A structure with fields such as `gradRasterTime`. | `mr.opts()` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `grads` | cell | Cell array containing the two resulting gradient structures. If nargout == 1, it returns a single cell array with both gradients. If nargout > 1, it returns each gradient separately. |

## Examples

```matlab
% Example 1: Split gradient for slice selection timing
gz_parts = mr.splitGradientAt(gz, rf.delay + rf.t(end));
gz_parts(1).delay = mr.calcDuration(gzReph);
gz_1 = mr.addGradients({gzReph, gz_parts(1)}, 'system', sys);

% Example 2: Split readout gradient at ADC end for optimal timing
gx_parts = mr.splitGradientAt(gx, ceil((adc.dwell*adc.numSamples+adc.delay+adc.deadTime)/sys.gradRasterTime)*sys.gradRasterTime);
gx_parts(1).delay = mr.calcDuration(gxPre);
gx_1 = mr.addGradients({gxPre, gx_parts(1)}, 'system', sys);

% Example 3: Split blip gradient for EPI sequence
gyBlip_parts = mr.splitGradientAt(gyBlip, blip_dur/2, sys);
[gyBlip_up, gyBlip_down, ~] = mr.align('right', gyBlip_parts(1), 'left', gyBlip_parts(2), gx);
```

## See Also

[addGradients](addGradients.md), [makeExtendedTrapezoid](makeExtendedTrapezoid.md), [makeTrapezoid](makeTrapezoid.md), [Sequence.addBlock](addBlock.md), [mr.opts](opts.md)
