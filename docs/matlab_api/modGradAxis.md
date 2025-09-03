# modGradAxis

This function modifies the amplitude of gradient events along a specified axis ('x', 'y', or 'z') within a Pulseq sequence.  It scales all gradient events on the selected axis by a given modifier.  It handles both single- and multi-point gradient events.  The function operates on the gradient events that have already been added to the sequence object's gradient library.

## Syntax

```matlab
function modGradAxis(axis,modifier)
```

## Calling Pattern

```matlab
seq.modGradAxis(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `axis` | char | The axis ('x', 'y', or 'z') along which to modify the gradient amplitudes. | `'x'` |  |
| `modifier` | double | The scaling factor applied to the gradient amplitudes on the specified axis.  A value of -1 inverts the gradients. | `-1` |  |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `trajectory_delay` | double | `0` | This parameter seems unused in the provided code excerpt.  Its purpose is unclear. (Units: seconds) | `0.001` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `none` | none | This function modifies the sequence object in place and does not return any values. |

## Examples

```matlab
seq.modGradAxis('x', 2); % Doubles the amplitude of all x-gradients
seq.modGradAxis('y', -1); % Inverts the amplitude of all y-gradients
```

## See Also

[flipGradAxis](flipGradAxis.md)
