# normalize

This function normalizes a quaternion or an array of quaternions.  It scales each quaternion to have a unit norm (magnitude of 1).  Quaternions with a norm of zero are left unchanged.

## Syntax

```matlab
function q = normalize(q)
```

## Calling Pattern

```matlab
mr.aux.quat.normalize(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `q` | double | A quaternion or an array of quaternions. Each quaternion is represented as a row vector of four elements [w, x, y, z]. | `[0.707, 0, 0, 0.707]` |  |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `q` | double | The normalized quaternion or array of quaternions.  Each quaternion will have a magnitude of approximately 1 (or remain unchanged if the original magnitude was zero). |

## Examples

```matlab
q_normalized = mr.aux.quat.normalize([1, 2, 3, 4]);
q_array_normalized = mr.aux.quat.normalize([0.707, 0, 0, 0.707; 0, 0.707, 0, 0.707]);
```
