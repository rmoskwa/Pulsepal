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
% TE delay in gradient echo sequence
seq.addBlock(mr.makeDelay(delayTE));

% Preparation delay for steady-state sequences
prepDelay = mr.makeDelay(round((TR/2 - mr.calcDuration(gz_1))/sys.gradRasterTime)*sys.gradRasterTime);

% Inversion recovery delay
seq.addBlock(mr.makeDelay(TIdelay), gslSp);
```

## See Also

[Sequence.addBlock](addBlock.md)
