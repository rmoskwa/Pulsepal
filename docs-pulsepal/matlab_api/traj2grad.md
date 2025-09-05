# traj2grad

Converts a k-space trajectory into gradient waveforms using finite differences.  The input trajectory is assumed to be in units of 1/m and sampled at the raster edges (unless otherwise specified). The function calculates both the gradient waveform and the slew rate. It offers options for handling the first gradient step and for a more conservative slew rate estimate.

## Syntax

```matlab
function [g sr]=traj2grad(k,varargin)
```

## Calling Pattern

```matlab
mr.traj2grad(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `k` | double | The k-space trajectory.  The size of k is [nChannel nTime], where nChannel is the number of channels and nTime is the number of time points. | `[0.1 0.2 0.3; 0.4 0.5 0.6]` | 1/m |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `first` | double | `[]` | The initial gradient value. If not provided, it defaults to zero. (Units: Hz/m) | `[10; 20]` |
| `firstGradStepHalfRaster` | logical | `true` | Specifies whether the first gradient step should be considered half a raster time.  Affects the slew rate calculation. | `true` |
| `conservativeSlewEstimate` | logical | `false` | If true, uses a conservative estimate for the slew rate, taking the maximum absolute slew rate between adjacent gradient points. If false, uses the average of adjacent slew rates. | `false` |
| `system` | struct | `[]` | A structure containing system parameters. If not provided, it defaults to mr.opts().  Likely contains parameters such as gradRasterTime. | `mr.opts()` |
| `RasterTime` | double | `[]` | The time duration of one raster time unit.  If not provided, it defaults to the gradRasterTime from the 'system' parameter (or mr.opts().gradRasterTime if 'system' is not provided). (Units: seconds) | `0.000004` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `g` | double | The calculated gradient waveform in Hz/m. |
| `sr` | double | The calculated slew rate in Hz/m/s. |

## Examples

```matlab
g = mr.traj2grad(k);
g = mr.traj2grad(k, 'RasterTime', 0.000004);
[g, sr] = mr.traj2grad(k, 'system', mr.opts('maxSlewRate', 150), 'firstGradStepHalfRaster', false);
```

## See Also

[mr.opts](opts.md), [Sequence.makeArbitraryGrad](makeArbitraryGrad.md)
