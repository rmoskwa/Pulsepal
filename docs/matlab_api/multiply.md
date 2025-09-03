# multiply

This function calculates the product of two quaternions.  A quaternion is represented as a 1x4 vector, where the first element is the scalar part and the remaining three elements represent the vector part. The function can handle single quaternions (1x4 vectors) or collections of quaternions (Nx4 matrices).

## Syntax

```matlab
function qout = multiply(q1,q2)
```

## Calling Pattern

```matlab
mr.aux.quat.multiply(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `q1` | double | The first quaternion or a collection of quaternions.  Each quaternion is represented as a 1x4 vector or row in an Nx4 matrix. | `[1, 0, 0, 0]` |  |
| `q2` | double | The second quaternion or a collection of quaternions.  Must have the same number of rows as q1 if q1 is an Nx4 matrix. Each quaternion is represented as a 1x4 vector or row in an Nx4 matrix. | `[0, 1, 0, 0]` |  |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `qout` | double | The resulting quaternion or collection of quaternions representing the product of q1 and q2. The output will have the same dimensions as the input quaternions. |

## Examples

```matlab
qout = mr.aux.quat.multiply([1, 0, 0, 0], [0, 1, 0, 0]);
qout = mr.aux.quat.multiply([1, 0, 0, 0; 0, 1, 0, 0], [0, 1, 0, 0; 0, 0, 1, 0]);
```
