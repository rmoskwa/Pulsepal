# splitGradient

This function decomposes a trapezoidal gradient into its constituent parts: slew-up, flat-top, and slew-down.  It takes a trapezoidal gradient structure as input and returns three separate extended trapezoid gradient structures representing the individual components. The delays within these components are adjusted to ensure that when added together using `addGradients`, the resulting gradient is equivalent to the original input gradient.

## Syntax

```matlab
function [grads] = splitGradient(grad, varargin)
```

## Calling Pattern

```matlab
mr.splitGradient(...)
mr.splitGradient('ParameterName', value, ...)
```

## Parameters

### Required Parameters

| Parameter Name | Value Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `grad` | struct | A structure representing a trapezoidal gradient.  This structure must contain fields defining the gradient's amplitude, rise time, flat time, fall time, delay, and channel ('x','y','z').  It should be created using functions such as `mr.makeTrapezoid`. | `mr.makeTrapezoid('x', 10, 0.001, 0.002, 0.001, 0)` |  |

### Name-Value Pair Arguments
| Parameter Name (string) | Value Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `system` | struct | `[]` | A structure containing system parameters (e.g., gradient raster time). If omitted, default system parameters are used from `mr.opts()`. Valid values: A valid mr.opts structure | `mr.opts()` |

## Returns

| Output | Value Type | Description |
|--------|------|-------------|
| `grads` | struct | An array of three extended trapezoid gradient structures: slew-up, flat-top, and slew-down gradients. |

## Examples

```matlab
[grads] = mr.splitGradient(mr.makeTrapezoid('x', 40, 0.001, 0.002, 0.001, 0), mr.opts('gradRasterTime', 0.0001));
```

## See Also

[splitGradientAt](splitGradientAt.md), [makeExtendedTrapezoid](makeExtendedTrapezoid.md), [makeTrapezoid](makeTrapezoid.md), [Sequence.addBlock](addBlock.md), [mr.opts](opts.md)
