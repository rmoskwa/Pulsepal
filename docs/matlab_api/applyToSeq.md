# applyToSeq

This helper function applies a transformation defined by an object `obj` to a Pulseq sequence `seq`. It iterates through blocks of the sequence within a specified range and applies the transformation defined in `obj.applyToBlock` to each block.  It offers the option to modify the input sequence (`sameSeq = false`) or to perform the transformation in-place (`sameSeq = true`).

## Syntax

```matlab
function seq2 = applyToSeq(varargin)
```

## Calling Pattern

```matlab
tra.applyToSeq(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `seq` | Sequence | The Pulseq sequence object to apply the transformation to | `mySequence` |  |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `sameSeq` | logical | `false` | Specifies whether to modify the input sequence in-place. If true, the transformation is applied directly to the input sequence `seq`. If false (default), a new sequence `seq2` is created and the transformation is applied to this new sequence. Valid values: true, false | `true` |
| `blockRange` | numeric array | `[1 inf]` | A two-element array specifying the range of blocks to process. The first element is the starting block index, and the second element is the ending block index.  If the second element is Inf, all blocks from the starting index to the end of the sequence are processed. Valid values: [startIndex, endIndex] where startIndex and endIndex are positive integers, and endIndex can be Inf | `[5, 10]` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `seq2` | struct | A Pulseq sequence. If `sameSeq` is true, this is the same as the input `seq`, but modified in place. If `sameSeq` is false, this is a new Pulseq sequence containing the transformed blocks. |

## Examples

```matlab
tra.applyToSeq(mySequence);
tra.applyToSeq(mySequence, 'sameSeq', true, 'blockRange', [1, 10]);
```

## See Also

[applyToBlock](applyToBlock.md), [getBlock](getBlock.md)
