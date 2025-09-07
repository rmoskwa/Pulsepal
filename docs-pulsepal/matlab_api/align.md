# align

This function aligns the objects within a Pulseq block based on a specified alignment type ('left', 'center', or 'right').  It considers pre-existing delays within the objects, calculates the total block duration, and then adjusts object delays to achieve the desired alignment.  Optionally, a predefined block duration can be provided as input; the function will then check if the total object duration exceeds this limit and raise an error if it does.

## Syntax

```matlab
function [varargout] = align(varargin)
```

## Calling Pattern

```matlab
mr.align(...)
```

## Parameters


### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `align_spec` | char |  | Specifies the alignment of objects within the block.  Possible values are 'left', 'center', and 'right'. Valid values: 'left', 'center', 'right' | `'left'` |
| `obj` | struct |  | Pulseq object to be aligned. Multiple objects can be specified. Valid values: Pulseq object | `myPulseqObject` |
| `required_duration` | double |  | Optional. Specifies the desired duration of the block. If provided, the function will check if the total duration of the objects exceeds this value. Valid values: positive number (Units: seconds) | `0.01` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `objects` | cell | Cell array containing the aligned Pulseq objects. |
| `required_duration` | double | Optional return value.  Returns the required duration of the block (if this was specified as an input). |

## Examples

```matlab
% Align EPI blip gradients with readout gradient timing
[gy_blipup, gy_blipdown, ~] = mr.align('right', gy_parts(1), 'left', gy_parts(2), gx);

% Right-align prephasing gradients before readout
[gxPre, gyPre, gzReph] = mr.align('right', gxPre, 'left', gyPre, gzReph);

% Align RF pulse with slice-select gradient
[rf, ~] = mr.align('right', rf, gz_1);
```

## See Also

[Sequence.addBlock](addBlock.md)
