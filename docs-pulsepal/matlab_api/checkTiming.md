# checkTiming

This function checks the timing and other parameters of all blocks and objects within a Pulseq sequence.  It verifies consistency between stored block durations and calculated durations, ensures alignment with the blockDurationRaster, and checks for dead times in RF and ADC events. The function modifies the sequence object by adding a 'TotalDuration' field and returns a boolean indicating whether all checks passed, along with a detailed error report if any checks failed.

## Syntax

```matlab
function [is_ok, errorReport]=checkTiming()
```

## Calling Pattern

```matlab
seq.checkTiming(...)
```

## Parameters


### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `trajectory_delay` | double | `0` | Optional parameter that seems to be unused in the provided code excerpt.  It might be related to trajectory delays but its usage is not apparent here. Valid values: >= 0 (Units: seconds) | `0.001` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `is_ok` | logical | A boolean value indicating whether all timing checks passed (true) or not (false). |
| `errorReport` | cell | A cell array of strings containing detailed error messages if any timing checks failed.  If all checks pass, this will be an empty cell array. |

## Examples

```matlab
[is_ok, errorReport] = seq.checkTiming();
```

## See Also

[mr.checkTiming](checkTiming.md), [Sequence.getBlock](getBlock.md)
