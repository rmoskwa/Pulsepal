# findFlank

This function finds the x-coordinate of the left flank of a given function f. It determines the first x value where the absolute value of f exceeds a specified fraction (c) of the maximum absolute value of f.  Linear interpolation is used if the identified point is not the first element in the x vector.

## Syntax

```matlab
function xf=findFlank(x,f,c)
```

## Calling Pattern

```matlab
mr.aux.findFlank(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `x` | double | A vector of x-coordinates corresponding to the function values in f. | `[0:0.01:1]` |  |
| `f` | double | A vector of function values corresponding to the x-coordinates in x. | `sin(2*pi*x)` |  |
| `c` | double | A scaling factor (0 < c < 1) that determines the threshold for identifying the flank. The threshold is c * max(abs(f)). | `0.1` |  |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `xf` | double | The x-coordinate of the left flank of the function f. This is the first x value where abs(f) > c * max(abs(f)). Linear interpolation is used to refine the result if necessary. |

## Examples

```matlab
xf = mr.aux.findFlank([0:0.01:1], sin(2*pi*[0:0.01:1]), 0.5);
```
