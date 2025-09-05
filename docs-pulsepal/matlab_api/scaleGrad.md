# scaleGrad

Scales a gradient waveform by a given scalar value.  It handles both trapezoidal and arbitrary waveforms. Optionally, it checks the scaled gradient against system limits (maximum gradient amplitude and slew rate) to prevent exceeding hardware capabilities.

## Syntax

```matlab
function [grad] = scaleGrad(grad, scale, system)
```

## Calling Pattern

```matlab
mr.scaleGrad(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `grad` | struct | Structure containing the gradient waveform to be scaled. For trapezoidal gradients, it contains fields like 'amplitude', 'area', 'flatArea', 'riseTime', and 'fallTime'. For arbitrary waveforms, it contains fields like 'waveform' and 'tt'. | `{type: 'trap', amplitude: 10, area: 0.001, flatArea: 0.0005, riseTime: 0.001, fallTime: 0.001}` |  |
| `scale` | double | Scalar value by which to scale the gradient waveform. | `2.5` |  |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `system` | struct | `N/A` | Structure containing system limits.  Must contain fields 'maxGrad' (maximum gradient amplitude in Hz/m) and 'maxSlew' (maximum slew rate in Hz/m/s). Valid values: Must be a struct with 'maxGrad' and 'maxSlew' fields. | `{maxGrad: 40e6, maxSlew: 150e6}` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `grad` | struct | The scaled gradient waveform structure.  The 'id' field, if present, will be removed. |

## Examples

```matlab
[grad_scaled] = mr.scaleGrad(grad, 2);
[grad_scaled] = mr.scaleGrad(grad, 0.5, system);
```

## See Also

[makeTrapezoid](makeTrapezoid.md)
