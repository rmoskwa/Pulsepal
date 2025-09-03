# addRamps

This function adds ramp segments to a k-space trajectory to ensure smooth transitions between zero and the desired trajectory. It prevents violations of gradient and slew rate limits.  It can handle single or multiple k-space trajectories (provided as a cell array or matrix). It can also add corresponding zero-filled segments to an accompanying RF pulse.

## Syntax

```matlab
function varargout=addRamps(k,varargin)
```

## Calling Pattern

```matlab
mr.addRamps(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `k` | double|cell | The k-space trajectory.  Can be a numeric array (single trajectory) or a cell array of numeric arrays (multiple trajectories). Each column represents a time point. | `[1;2;3]` | 1/m |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `system` | struct | `[]` | A structure containing system limits (e.g., maxGrad, maxSlew). If empty, defaults to mr.opts(). | `mr.opts()` |
| `rf` | double | `[]` | An RF pulse shape. If provided, segments of zeros are added to match the duration of the added ramps. | `[0.5 0.5 0.5]` |
| `maxGrad` | double | `0` | Maximum gradient amplitude. Overrides the value in the 'system' structure if greater than 0. (Units: Hz/m) | `30e6` |
| `maxSlew` | double | `0` | Maximum gradient slew rate. Overrides the value in the 'system' structure if greater than 0. (Units: Hz/m/s) | `100e6` |
| `gradOversampling` | logical | `false` | Logical flag indicating whether gradient oversampling is used during ramp calculation. | `true` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `varargout` | double|cell | The k-space trajectory with added ramps.  Returns a numeric array if input k was numeric; returns a cell array if input k was a cell array.  If 'rf' is provided, it also returns the extended RF pulse. |

## Examples

```matlab
k_space = [1;2;3];
[k_space_with_ramps] = mr.addRamps(k_space, 'maxGrad', 30e6, 'maxSlew', 100e6);
k_space = {[1;2;3],[4;5;6]};
[k_space_with_ramps1, k_space_with_ramps2] = mr.addRamps(k_space, 'maxGrad', 30e6, 'maxSlew', 100e6);
k_space = [1;2;3]; rf_pulse = [0.5 0.5 0.5];
[k_space_with_ramps, rf_with_zeros] = mr.addRamps(k_space, 'rf', rf_pulse, 'maxGrad', 30e6, 'maxSlew', 100e6);
```

## See Also

[mr.opts](opts.md), [mr.calcRamp](calcRamp.md), [Sequence.makeArbitraryGrad](makeArbitraryGrad.md)
