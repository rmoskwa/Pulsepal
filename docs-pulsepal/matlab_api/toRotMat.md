# toRotMat

This function converts a normalized quaternion into its corresponding 3x3 rotation matrix.  The quaternion must be a 4-element vector.

## Syntax

```matlab
function r = toRotMat(q)
```

## Calling Pattern

```matlab
mr.aux.quat.toRotMat(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `q` | double | A 4-element vector representing a normalized quaternion [q1, q2, q3, q4]. | `[0.707, 0, 0, 0.707]` |  |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `r` | double | A 3x3 rotation matrix corresponding to the input quaternion. |

## Examples

```matlab
r = mr.aux.quat.toRotMat([0.707, 0, 0, 0.707]);
```
