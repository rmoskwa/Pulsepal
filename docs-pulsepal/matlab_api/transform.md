# transform

Creates a transformed copy of a Pulseq sequence by applying a rotation, translation, or a 4x4 homogeneous transformation matrix.  It modifies the sequence's gradient waveforms and calculates updated k-space phase information based on the transformation.

## Syntax

```matlab
function [seq2, gw_pp]= transform(seq, varargin)
```

## Calling Pattern

```matlab
mr.transform(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `seq` | struct | The input Pulseq sequence structure to be transformed. | `mySequence` |  |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `rotation` | double | `[]` | A 3x3 rotation matrix to apply to the sequence.  Cannot be used with 'transform'. Valid values: 3x3 numeric matrix | `[1 0 0; 0 1 0; 0 0 1]` |
| `translation` | double | `[]` | A 1x3 translation vector in Pulseq logical coordinates [x, y, z]. Cannot be used with 'transform'. Valid values: 1x3 numeric vector (Units: meters) | `[0.01 0 0]` |
| `offset` | double | `[]` | A translation vector in Pulseq logical coordinates [x, y, z]. Valid values: 1x3 numeric vector (Units: meters) | `[0.02, 0, 0]` |
| `transform` | double | `[]` | A 4x4 homogeneous transformation matrix containing both rotation and translation (in lab coordinates). Cannot be used with 'rotation' or 'translation'. Valid values: 4x4 numeric matrix | `[1 0 0 0.01; 0 1 0 0; 0 0 1 0; 0 0 0 1]` |
| `system` | struct | `[]` | Optional MR system description. If not provided, system properties from the input sequence are inherited. | `mr.opts('maxGrad', 40, 'maxSlew', 130)` |
| `sameSeq` | logical | `false` | If true, the output sequence will be a pointer to the input sequence; otherwise, a copy is created. | `true` |
| `blockRange` | double | `[1 inf]` | Specifies the range of blocks in the sequence to process.  The second value can be 'inf' for all blocks. Valid values: 1x2 numeric vector | `[10, 20]` |
| `gw_pp` | cell | `{}` | Optional pre-calculated gradient piecewise polynomial (k-space) data. If provided, the function will reuse this data instead of recalculating it. | `{}` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `seq2` | struct | The transformed Pulseq sequence. |
| `gw_pp` | cell | Gradient piecewise polynomial (k-space) data for the transformed sequence. |

## Examples

```matlab
[seq2, gw_pp] = mr.transform(mySequence, 'rotation', rotMatrix, 'offset', [0.01, 0, 0]);
[seq2, gw_pp] = mr.transform(mySequence, 'transform', homogeneousTransformMatrix);
```

## See Also

[mr.rotate](rotate.md), [mr.rotate3D](rotate3D.md)
