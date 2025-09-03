# fromRotMat

This function converts a 3x3 rotation matrix into a normalized quaternion.  It handles potential rounding errors and cases where the rotation matrix might be close to a zero or identity matrix.

## Syntax

```matlab
function q = fromRotMat(R)
```

## Calling Pattern

```matlab
mr.aux.quat.fromRotMat(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `R` | double | A 3x3 rotation matrix. | `[1 0 0; 0 1 0; 0 0 1]` |  |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `q` | double | A 4-element normalized quaternion representing the rotation. The order is [qs, qx, qy, qz]. |

## Examples

```matlab
q = mr.aux.quat.fromRotMat([1 0 0; 0 1 0; 0 0 1]);
```

## See Also

[mr.aux.quat.normalize](normalize.md)
