# setBlock

This function adds or replaces a block of events within a Pulseq sequence. It accepts events in uncompressed form and stores them in the sequence's internal, compressed, non-redundant libraries.  It handles different input formats: a pre-defined block structure, individual events, or a duration with subsequent events.  Error checking ensures that event durations do not exceed a specified duration when provided.

## Syntax

```matlab
function setBlock(index, varargin)
```

## Calling Pattern

```matlab
seq.setBlock(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `index` | double | The index specifying the location (block number) within the sequence where the new block should be added or replaced.  Indexing starts at 1. | `1` |  |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `trajectory_delay` | double | `0` | Delay added to the trajectory of the block.  This is relevant to gradient and RF events in the block. (Units: seconds) | `0.001` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `none` | none | This function modifies the Pulseq sequence object in place. It does not return any value. |

## Examples

```matlab
seq.setBlock(1, rf_event1, grad_event1); % Adds a block with an RF and a gradient event at index 1
seq.setBlock(2, block_struct); % Replaces block at index 2 with a given block structure
seq.setBlock(3, 0.01, rf_event2, grad_event2); % Creates a block with duration 0.01 seconds at index 3
```

## See Also

[getBlock](getBlock.md), [addBlock](addBlock.md)
