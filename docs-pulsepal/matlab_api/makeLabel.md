# makeLabel

Creates a label event for use in Pulseq sequences.  This function generates a structure defining a label that can be added to a Pulseq sequence using `Sequence.addBlock`.  Labels allow for controlling the execution flow and parameter values within the sequence based on various counters, flags, and control signals.

## Syntax

```matlab
function out = makeLabel(type, label, value)
```

## Calling Pattern

```matlab
mr.makeLabel(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `type` | char | Specifies the type of label operation: 'SET' to set a label's value or 'INC' to increment it. | `'SET'` |  |
| `label` | char | Specifies the name of the label.  This should be one of the supported labels (counters, flags, or control signals) returned by `mr.getSupportedLabels()`. | `'REP'` |  |
| `value` | double|logical | Specifies the value for the label.  For counters, this is a numeric value (increments can be negative). For flags, this is a logical value (true/false). | `10 or true` |  |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `out` | struct | A structure containing the label information.  It has fields 'type' ('labelset' or 'labelinc'), 'label' (the label name), and 'value' (the label value). |

## Examples

```matlab
% Initialize slice counter at start of multi-slice sequence
seq.addBlock(mr.makeLabel('SET','SLC', 0));

% Increment line counter for k-space encoding
seq.addBlock(mr.makeLabel('INC','LIN', 1));

% Set navigation echo flag for reference scan
seq.addBlock(mr.makeLabel('SET','NAV',1), mr.makeLabel('SET','LIN', floor(Ny/2)));
```

## See Also

[Sequence.addBlock](addBlock.md), [mr.getSupportedLabels](getSupportedLabels.md)
