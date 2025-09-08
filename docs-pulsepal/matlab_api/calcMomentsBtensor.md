# calcMomentsBtensor

Calculates the magnetic field (B) and its first three moments (m1, m2, m3) from a Pulseq sequence object.  The function processes gradient waveforms to compute these values, which are useful for characterizing the magnetic field variations in MRI experiments. It handles multiple repetitions (readouts) and allows for skipping initial dummy scans.

## Syntax

```matlab
function [B, m1, m2, m3] = calcMomentsBtensor(varargin)
```

## Calling Pattern

```matlab
seq.calcMomentsBtensor(...)
seq.calcMomentsBtensor('ParameterName', value, ...)
```

## Parameters

This function accepts both positional parameters and name-value pairs.

### Name-Value Pair Arguments
| Parameter Name (string) | Value Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `'calcB'` | boolean | `true` | Flag to calculate the magnetic field B tensor. If false, B will not be computed and returned. Valid values: true, false | `true` |
| `'calcm1'` | boolean | `false` | Flag to calculate the first moment (m1) of the magnetic field. If false, m1 will not be computed and returned. Valid values: true, false | `true` |
| `'calcm2'` | boolean | `false` | Flag to calculate the second moment (m2) of the magnetic field. If false, m2 will not be computed and returned. Valid values: true, false | `true` |
| `'calcm3'` | boolean | `false` | Flag to calculate the third moment (m3) of the magnetic field. If false, m3 will not be computed and returned. Valid values: true, false | `true` |
| `'Ndummy'` | integer | `0` | The number of dummy scans (initial scans to skip) in the sequence. Valid values: 0, 1, 2, ... | `2` |

## Returns

| Output | Value Type | Description |
|--------|------|-------------|
| `B` | 3D array | A 3D array representing the magnetic field tensor. Dimensions are [repetition, 3, 3].  Returned only if calcB is true. |
| `m1` | 2D array | A 2D array representing the first moment of the magnetic field. Dimensions are [repetition, 3]. Returned only if calcm1 is true. |
| `m2` | 2D array | A 2D array representing the second moment of the magnetic field. Dimensions are [repetition, 3]. Returned only if calcm2 is true. |
| `m3` | 2D array | A 2D array representing the third moment of the magnetic field. Dimensions are [repetition, 3]. Returned only if calcm3 is true. |

## Examples

```matlab
[B, m1, m2, m3] = seq.calcMomentsBtensor('calcm1', true, 'Ndummy', 2);
```
