# rotate3D

Rotates gradient objects within a Pulseq block using a provided rotation matrix or quaternion.  Non-gradient objects are unaffected.  It accepts either a 3x3 rotation matrix or a unit quaternion (scalar component first). An optional 'system' parameter can specify system limits. The function returns either a cell array or a list of rotated objects, suitable for use with `seq.addBlock()`.

## Syntax

```matlab
function [varargout] = rotate3D(rotation, varargin)
```

## Calling Pattern

```matlab
mr.rotate3D(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `rotation` | double | A 3x3 rotation matrix or a 4-element unit quaternion (scalar component first) specifying the rotation to be applied. | `[1 0 0; 0 1 0; 0 0 1] or [1 0 0 0]` |  |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `system` | struct | `[]` | A structure containing system limits (e.g., `gradRasterTime`, etc.).  Must be placed as the first or last optional argument, preceded or followed by the keyword 'system'. Valid values: Must contain at least 'gradRasterTime' field. | `struct('gradRasterTime', 4e-6)` |
| `obj` | struct|cell | `[]` | One or more Pulseq gradient objects to be rotated.  Can be a cell array of multiple objects or a sequence of objects. Valid values: Pulseq gradient objects or a cell array of them. | `{... gradient object ...}` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `varargout` | cell|struct | Either a cell array of rotated objects (if one output argument is requested) or a list of individual rotated objects (if multiple output arguments are requested). |

## Examples

```matlab
[rotatedObjects] = mr.rotate3D([1 0 0; 0 1 0; 0 0 1], grad_x, grad_y, grad_z);
[rotatedObjects] = mr.rotate3D([0 0 1; 1 0 0; 0 1 0], grad_x, grad_y, 'system', systemStruct);
[rotatedObjects] = mr.rotate3D([1 0 0 0], {grad_x, grad_y, grad_z});
```

## See Also

[mr.rotate](rotate.md), [Sequence.addBlock](addBlock.md), [mr.aux.quat.toRotMat](toRotMat.md)
