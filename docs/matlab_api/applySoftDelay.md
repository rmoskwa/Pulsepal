# applySoftDelay

This function adjusts the durations of blocks within a Pulseq sequence based on user-specified soft delays.  It takes key-value pairs as input, where the keys are string identifiers (hints) for soft delays (e.g., 'TE', 'TR') and the values are the desired durations in seconds.  The function iterates through the sequence's blocks, identifying those with matching soft delay hints. It then updates the block durations accordingly, ensuring consistency between the numeric and string IDs associated with each soft delay.  Soft delays not specified in the input are left unchanged.

## Syntax

```matlab
function applySoftDelay(varargin)
```

## Calling Pattern

```matlab
seq.applySoftDelay(...)
```

## Parameters


### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `blockRange` | double | `[1 inf]` | Specifies the range of blocks to modify. Defaults to applying soft delays to all blocks. Valid values: A two-element vector defining the start and end indices of the blocks to process.  Must contain positive integers. | `[5 10]` |
| `channelWeights` | double | `[1 1 1]` | A vector of weights applied to the soft delays on different channels.  Not directly used in the provided excerpt but likely intended for applying different delays to different channels. Valid values: A three-element vector representing weights for x, y, and z channels. | `[0.8 1 1.2]` |
| `onlyProduceSoundData` | logical | `false` | A flag indicating whether to only produce sound data. Not directly used in the provided excerpt, but likely related to optional data generation. Valid values: true or false | `true` |

## Examples

```matlab
seq.applySoftDelay('TE', 40e-3);
seq.applySoftDelay('TE', 50e-3, 'TR', 2);
```
