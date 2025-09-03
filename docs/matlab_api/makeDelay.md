# makeDelay

Creates a delay event for use in a Pulseq sequence.  This function generates a structure representing a delay block with a specified duration.

## Syntax

```matlab
function del = makeDelay(delay)
```

## Calling Pattern

```matlab
mr.makeDelay(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `delay` | double | The duration of the delay in seconds. | `0.005` | seconds |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `del` | struct | A structure representing the delay event.  This structure contains the field 'type' with value 'delay' and the field 'delay' containing the specified delay duration. |

## Examples

```matlab
delayEvent = mr.makeDelay(0.01); % Creates a 10ms delay
```

## See Also

[Sequence.addBlock](addBlock.md)
