# makeArbitraryGrad

Creates a gradient event with an arbitrary waveform.  The function ensures the waveform adheres to the specified gradient hardware constraints (maximum slew rate and amplitude). It allows for oversampling and extrapolation to handle waveform edges.

## Syntax

```matlab
function grad=makeArbitraryGrad(channel,varargin)
```

## Calling Pattern

```matlab
mr.makeArbitraryGrad(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `channel` | char | The gradient channel ('x', 'y', or 'z') for the waveform. | `'x'` |  |
| `waveform` | double | A vector representing the desired gradient waveform amplitude at each time point. | `[0.1, 0.2, 0.3, 0.2, 0.1]` | Hz/m |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `system` | struct | `[]` | A structure containing system parameters (e.g., maxSlew, maxGrad, gradRasterTime). If empty, defaults to mr.opts(). | `mr.opts()` |
| `oversampling` | logical | `false` | Specifies whether oversampling is used. If true, the waveform is sampled at twice the resolution.  | `true` |
| `maxGrad` | double | `0` | The maximum allowed gradient amplitude. If 0, the system's maxGrad is used. (Units: Hz/m) | `1000` |
| `maxSlew` | double | `0` | The maximum allowed gradient slew rate. If 0, the system's maxSlew is used. (Units: Hz/m/s) | `100000` |
| `delay` | double | `0` | Delay before the gradient waveform starts. (Units: seconds) | `0.001` |
| `first` | double | `NaN` | The gradient amplitude at the very beginning (before the first sample in 'waveform'). If NaN, it's extrapolated. (Units: Hz/m) | `0` |
| `last` | double | `NaN` | The gradient amplitude at the very end (after the last sample in 'waveform'). If NaN, it's extrapolated. (Units: Hz/m) | `0` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `grad` | struct | A structure containing the gradient event details: type, channel, waveform, delay, area, and tt (time points). |

## Examples

```matlab
g = mr.makeArbitraryGrad('x', [0 1 0], 'maxGrad', 2000, 'maxSlew', 200000);
g = mr.makeArbitraryGrad('y', [0:0.1:1 1:-0.1:0], mr.opts(), true);
```

## See Also

[Sequence.addBlock](addBlock.md), [mr.opts](opts.md)
