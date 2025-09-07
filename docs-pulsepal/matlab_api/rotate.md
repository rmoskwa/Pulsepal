# rotate

Rotates a 3D vector by a given unit quaternion.  The function efficiently implements quaternion rotation using explicit formulas to avoid the overhead of general quaternion multiplication.

## Syntax

```matlab
function r = rotate(q,v)
```

## Calling Pattern

```matlab
mr.rotate(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `q` | double | A unit quaternion representing the rotation.  It should be a 4-element vector or an N x 4 matrix where each row represents a quaternion.  The quaternion should be in the form [scalar, vector_x, vector_y, vector_z]. | `[1, 0, 0, 0]` |  |
| `v` | double | A 3D vector or an N x 3 matrix of 3D vectors to be rotated. Each column represents a component (x, y, z) of the vector. | `[1; 2; 3]` |  |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `r` | double | The rotated vector(s).  If 'v' is a 3D vector, 'r' is a 3D vector. If 'v' is an N x 3 matrix, 'r' is an N x 3 matrix where each row represents a rotated vector. |

## Examples

```matlab
% Rotate gradient and ADC events for spiral imaging
seq.addBlock(mr.rotate('z', phi, gzReph, gx, gy, adc));

% Rotate spoiler gradients by angle phi around z-axis
seq.addBlock(mr.rotate('z', phi, gx_spoil, gy_spoil, gz_spoil));

% Rotate diffusion gradient for 3-axis encoding
Gr = mr.rotate('z', azimuth, mr.rotate('y', polar, gDiff));
```
