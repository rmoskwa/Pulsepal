# findBlockByTime

This helper function finds the index of the block in a Pulseq sequence that contains a given time point.  It takes a sequence object and a time value as input and returns the index of the block that encompasses that time. The function iterates through the cumulative sum of block durations until it finds the block containing the specified time.  It handles cases where the time is beyond the sequence duration by returning an empty array.

## Syntax

```matlab
function iB=findBlockByTime(t)
```

## Calling Pattern

```matlab
seq.findBlockByTime(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `t` | double | The time point (in seconds) to search for within the sequence blocks. | `0.015` | seconds |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `trajectory_delay` | double | `0` | This parameter appears in the original code but is not used in the provided excerpt.  It's likely intended to account for delays in trajectory calculation but is unused in this version of the function. (Units: seconds) | `0.001` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `iB` | double | The index of the block in the sequence object (obj.blockDurations) that contains the specified time point 't'. Returns an empty array if 't' is beyond the sequence duration. |

## Examples

```matlab
iB = seq.findBlockByTime(0.010);
```

## See Also

[Sequence](Sequence.md)
