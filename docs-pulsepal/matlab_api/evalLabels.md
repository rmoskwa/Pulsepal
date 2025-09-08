# evalLabels

This function evaluates the values of labels used within a Pulseq sequence. It can calculate the final label values at the end of the sequence or track their evolution across specified blocks.  It supports initializing label values and selectively evaluating labels based on the presence of ADCs or label manipulations.

## Syntax

```matlab
function labels = evalLabels(varargin)
```

## Calling Pattern

```matlab
seq.evalLabels(...)
seq.evalLabels('ParameterName', value, ...)
```

## Parameters

This function accepts both positional parameters and name-value pairs.

### Name-Value Pair Arguments
| Parameter Name (string) | Value Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `'blockRange'` | double | `[1 inf]` | Specifies the range of blocks to evaluate labels for.  The default evaluates all blocks. Valid values: A two-element numeric vector [first last], where 'first' and 'last' are the indices of the starting and ending blocks respectively. 'inf' can be used for the last element to denote the last block. | `[5 10]` |
| `'init'` | struct | `struct([]) ` | Provides initial values for labels. Useful for evaluating labels block-by-block, where results from the previous block are used as inputs for the next. | `struct('label1',10, 'label2',0)` |
| `'evolution'` | char | `'none'` | Specifies the level of detail for the label evolution output. Valid values: 'none', 'adc', 'label', 'blocks' | `'blocks'` |

## Returns

| Output | Value Type | Description |
|--------|------|-------------|
| `labels` | struct | A structure containing the evaluated label values. Field names correspond to the label names used in the sequence. |

## Examples

```matlab
% Evaluate label evolution during ADC events for debugging
lbls = seq.evalLabels('evolution', 'adc');
lbl_names = fieldnames(lbls);
figure; hold on;
for n = 1:length(lbl_names)
    plot(lbls.(lbl_names{n}));
end
legend(lbl_names(:));
title('evolution of labels/counters/flags');
xlabel('adc number');

% Basic label evaluation at sequence end
labels = seq.evalLabels();

% Evaluate labels over specific block range
labels = seq.evalLabels('blockRange', [10 20], 'evolution', 'adc');
```

## See Also

[getBlock](getBlock.md)
