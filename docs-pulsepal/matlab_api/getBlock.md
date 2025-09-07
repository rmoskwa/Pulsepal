# getBlock

Retrieves a specific block from a Pulseq sequence object.  The function decompresses all events and shapes within the requested block and returns it as a structure. It also handles the extraction and unpacking of optional extensions like triggers and labels associated with the block.

## Syntax

```matlab
function block = getBlock(index, addIDs)
```

## Calling Pattern

```matlab
seq.getBlock(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `index` | double | The index (integer) of the block to retrieve (1-based indexing) | `1` |  |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `addIDs` | logical | `false` | A boolean flag indicating whether to include IDs for triggers and labels in the returned block structure | `true` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `block` | struct | A structure containing the block's data.  This includes fields such as 'blockDuration', 'rf', 'gx', 'gy', 'gz', 'adc', and potentially 'trig' (for triggers) and other fields related to labels and other extensions if present in the raw block data. |

## Examples

```matlab
% Retrieve block for timing calculations
b = seq.getBlock(iB);
duration = mr.calcDuration(b);

% Get specific blocks for TR verification
assert(TR==(mr.calcDuration(seq.getBlock(5))+mr.calcDuration(seq.getBlock(6))));

% Retrieve block with ID information included
block = seq.getBlock(2, true);
```

## See Also

[setBlock](setBlock.md), [addBlock](addBlock.md)
