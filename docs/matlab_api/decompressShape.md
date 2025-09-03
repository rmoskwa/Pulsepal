# decompressShape

Decompresses a gradient or pulse shape that has been compressed using a run-length encoding scheme on its derivative.  The input shape is a structure containing the compressed waveform and the number of samples in the uncompressed waveform. The function reconstructs the original waveform by iteratively expanding the run-length encoded segments.

## Syntax

```matlab
function w = decompressShape(shape, forceDecompression)
```

## Calling Pattern

```matlab
mr.decompressShape(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `shape` | struct | Structure containing the compressed waveform data.  Must have fields 'num_samples' (number of samples in the uncompressed waveform) and 'data' (the compressed waveform data). | `{ 'num_samples': 1000, 'data': [1, 0, 2, 0, 0, 3, ... ] }` |  |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `forceDecompression` | logical | `false` | If true, forces decompression even if the input shape appears to be already uncompressed (i.e., the number of samples matches the length of the compressed data). Valid values: true, false | `true` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `w` | double | A vector containing the decompressed waveform. |

## Examples

```matlab
w = mr.decompressShape(shape);
w = mr.decompressShape(shape, true);
```

## See Also

[compressShape](compressShape.md)
