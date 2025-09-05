# calcRamp

Calculates a k-space trajectory connecting two given points (k0 and kend) while adhering to gradient and slew rate limits.  The function iteratively tries to find a solution with an increasing number of intermediate points until it finds a trajectory satisfying the constraints, or until a maximum number of points is reached. The resulting trajectory is not guaranteed to be the absolute shortest, but it's a reasonably short connection.

## Syntax

```matlab
function [kout, success] = calcRamp(k0,kend,varargin)
```

## Calling Pattern

```matlab
mr.calcRamp(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `k0` | double | Two preceding points in k-space.  Size is [3,2], representing the x, y, and z components of the two points. | `[ [0;0;0], [1;1;1] ]` | 1/m |
| `kend` | double | Two following points in k-space. Size is [3,2], representing the x, y, and z components of the two points. | `[ [2;2;2], [3;3;3] ]` | 1/m |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `system` | struct | `[]` | A structure containing system parameters. If empty, default system parameters are used.  Should contain fields like maxGrad and maxSlew. | `mr.opts()` |
| `MaxPoints` | double | `500` | Maximum number of k-space points allowed in the connecting trajectory. Limits the computation time. Valid values: >0 | `1000` |
| `maxGrad` | double | `0` | Maximum gradient strength. Can be a scalar (total vector gradient) or a 3x1 vector (per-coordinate limits). If 0, the value from the 'system' parameter is used. (Units: Hz/m) | `40e6` |
| `maxSlew` | double | `0` | Maximum slew rate. Can be a scalar (total vector slew rate) or a 3x1 vector (per-coordinate limits). If 0, the value from the 'system' parameter is used. (Units: Hz/(m*s)) | `200e6` |
| `gradOversampling` | logical | `false` | If true, the gradient raster time is halved. Affects the gradient discretization. | `true` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `kout` | double | Connecting k-space points (excluding k0 and kend). Size is [3, Nt], where Nt is the number of steps between k0 and kend. Units are 1/m. |
| `success` | logical | A flag indicating whether a solution was found (1) or not (0). |

## Examples

```matlab
[kout, success] = mr.calcRamp([ [0;0;0], [1;1;1] ], [ [2;2;2], [3;3;3] ], 'maxGrad', 40e6, 'maxSlew', 200e6)
```
