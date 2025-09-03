# evalLabels

This function evaluates the values of labels used within a Pulseq sequence. It can calculate the final label values at the end of the sequence or track their evolution across specified blocks.  It supports initializing label values and selectively evaluating labels based on the presence of ADCs or label manipulations.

## Syntax

```matlab
function labels = evalLabels(varargin)
```

## Calling Pattern

```matlab
seq.evalLabels(...)
```

## Parameters


### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `blockRange` | double | `[1 inf]` | Specifies the range of blocks to evaluate labels for.  The default evaluates all blocks. Valid values: A two-element numeric vector [first last], where 'first' and 'last' are the indices of the starting and ending blocks respectively. 'inf' can be used for the last element to denote the last block. | `[5 10]` |
| `init` | struct | `struct([]) ` | Provides initial values for labels. Useful for evaluating labels block-by-block, where results from the previous block are used as inputs for the next. | `struct('label1',10, 'label2',0)` |
| `evolution` | char | `'none'` | Specifies the level of detail for the label evolution output. Valid values: 'none', 'adc', 'label', 'blocks' | `'blocks'` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `labels` | struct | A structure containing the evaluated label values. Field names correspond to the label names used in the sequence. |

## Examples

```matlab
labels = seq.evalLabels();
labels = seq.evalLabels('blockRange', [10 20], 'evolution', 'adc');
labels = seq.evalLabels('init', struct('phase', pi/2));
```

## See Also

[getBlock](getBlock.md)
