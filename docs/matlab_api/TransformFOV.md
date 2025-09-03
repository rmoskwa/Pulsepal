# TransformFOV

This function transforms the field of view (FOV) of a Pulseq sequence by applying rotation, translation, and scaling operations.  It takes as input a set of Pulseq events (e.g., RF pulses, gradients, ADC events) and modifies their spatial coordinates according to the specified transformations. The function also handles an optional prior phase cycle and system parameters.

## Syntax

```matlab
function obj = TransformFOV(varargin)
```

## Calling Pattern

```matlab
tra = mr.TransformFOV(...)
```

## Parameters


### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `rotation` | double | `[]` | A 3x3 rotation matrix defining the rotation of the FOV.  An empty matrix indicates no rotation. (Units: radians) | `[1 0 0; 0 1 0; 0 0 1]` |
| `translation` | double | `[]` | A 1x3 vector representing the translation of the FOV. An empty matrix indicates no translation. (Units: meters) | `[0.01 0 0]` |
| `scale` | double | `[]` | A 1x3 vector defining scaling factors along each axis. An empty matrix indicates no scaling. | `[1 1 1]` |
| `prior_phase_cycle` | double | `0` | An integer representing a prior phase cycle to be applied before the transformation. Typically used for k-space trajectory calculations. | `0` |
| `high_accuracy` | logical | `false` | A logical flag indicating whether to use a high-accuracy transformation algorithm (currently commented out). | `true` |
| `system` | struct | `[]` | A structure containing system parameters (e.g., gradient limits, slew rates). If empty, default system parameters are used. | `mr.opts()` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `obj` | struct | A structure containing the transformation parameters and possibly the transformed sequence data. |

## Examples

```matlab
tra = mr.TransformFOV('rotation', [0 1; -1 0], 'translation', [0.01, 0, 0]);
tra = mr.TransformFOV('scale', [1.1 0.9 1], 'system', mr.opts('maxGrad', 40));
```

## See Also

[mr.opts](opts.md)
