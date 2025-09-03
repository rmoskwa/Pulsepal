# flipGradAxis

This helper function inverts all gradients along a specified axis or channel within a Pulseq sequence object.  It modifies existing gradient objects that have already been added to the sequence.

## Syntax

```matlab
function flipGradAxis(axis)
```

## Calling Pattern

```matlab
seq.flipGradAxis(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `axis` | char | The axis along which to invert the gradients ('x', 'y', or 'z'). | `'x'` |  |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `trajectory_delay` | double | `0` | This parameter is not used in the provided code snippet. It's likely an optional parameter intended for other functionality within the larger `Sequence` class. Valid values: >= 0 (Units: seconds) | `0.001` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `None` | None | This function modifies the sequence object in place and does not return any values. |

## Examples

```matlab
seq.flipGradAxis('x');
```

## See Also

[modGradAxis](modGradAxis.md)
