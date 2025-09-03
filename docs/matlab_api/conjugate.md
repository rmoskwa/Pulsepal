# conjugate

This function computes the conjugate of a quaternion or a collection of quaternions.  A quaternion is represented as a 1x4 vector [real, i, j, k], where the first element is the real part and the remaining three elements represent the imaginary components. The function handles single quaternions (1x4 vectors) and collections of quaternions (Nx4 matrices).

## Syntax

```matlab
function q = conjugate(q)
```

## Calling Pattern

```matlab
mr.aux.quat.conjugate(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `q` | double | A single quaternion (1x4 vector) or a collection of quaternions (Nx4 matrix). | `[1, 2, 3, 4]` |  |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `q` | double | The conjugate of the input quaternion or quaternions.  For a single quaternion, this is a 1x4 vector. For a collection of quaternions, this is an Nx4 matrix. |

## Examples

```matlab
q_conj = mr.aux.quat.conjugate([1, 2, 3, 4]);
q_conj_matrix = mr.aux.quat.conjugate([[1, 2, 3, 4]; [5, 6, 7, 8]]);
```
