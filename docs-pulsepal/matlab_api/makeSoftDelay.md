# makeSoftDelay

Creates a soft delay event for use with Pulseq sequences. This event modifies the duration of an empty (pure delay) block within a sequence, allowing for adjustments to timings like TE or TR.  The duration is calculated using the formula: dur = input / factor + offset.  This function is typically used in conjunction with `Sequence.addBlock()` and `Sequence.applySoftDelay()`.

## Syntax

```matlab
function sd = makeSoftDelay(varargin)
```

## Calling Pattern

```matlab
mr.makeSoftDelay(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `numID` | double | A numeric ID that uniquely identifies this soft delay event.  It's used to link this event to specific blocks within the sequence. | `1` |  |
| `hint` | char | A string hint associated with the soft delay event. This hint should be unique for each numID to distinguish between different types of delays. | `TE_adjust` |  |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `offset` | double | `0` | A constant offset added to the calculated delay duration.  This can be used to fine-tune the delay. Valid values: Any numeric value (positive or negative) (Units: seconds) | `0.001` |
| `factor` | double | `1` | A scaling factor applied to the input value before adding the offset.  This allows for scaling of the delay. Valid values: Any numeric value (positive or negative) | `2` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `sd` | struct | A structure containing the parameters of the created soft delay event. This structure is then used with `Sequence.applySoftDelay()` to apply the delay to the sequence. |

## Examples

```matlab
sd = mr.makeSoftDelay(1, 'TE_adjust', 'offset', 0.002, 'factor', 0.5);
sd = mr.makeSoftDelay(2, 'TR_adjust');
```

## See Also

[Sequence.addBlock](addBlock.md), [Sequence.applySoftDelay](applySoftDelay.md)
