# duration

Calculates the total duration of a Pulseq sequence and optionally returns the number of blocks and a count of events within each block.

## Syntax

```matlab
function [duration, numBlocks, eventCount]=duration()
```

## Calling Pattern

```matlab
seq.duration(...)
```

## Parameters

*No parameters*

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `duration` | double | The total duration of the sequence in seconds. |
| `numBlocks` | double | The number of blocks in the sequence. |
| `eventCount` | double | A vector representing the count of events (where an event is considered to be greater than 0 in the blockEvents) in each block. Only returned if requested (nargout > 2). |

## Examples

```matlab
% Get total sequence duration and block count
[total_duration, total_numBlocks] = seq.duration();

% Check TR per slice after adding blocks
if s==1
    TR_1slice = seq.duration; % note the actual TR per slice
end

% Calculate TR timing in sequence loop
if TR<=0
    TR = seq.duration;
end
```

## See Also

[checkTiming](checkTiming.md)
