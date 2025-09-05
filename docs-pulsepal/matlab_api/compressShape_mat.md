# compressShape_mat

Compresses a gradient or pulse waveform using a run-length encoding scheme applied to its derivative.  This efficiently represents constant and linear segments of the waveform, reducing storage requirements. The function returns a structure containing the compressed waveform data and the number of samples in the original uncompressed waveform.

## Syntax

```matlab
function s=compressShape_mat(w, forceCompression)
```

## Calling Pattern

```matlab
mr.compressShape_mat(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `w` | double | The uncompressed waveform data (gradient or pulse shape).  This is a vector of amplitude values. | `[1, 1, 1, 2, 2, 3, 3, 3, 2, 2, 1, 1]` |  |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `forceCompression` | logical | `false` | A logical flag indicating whether to force compression even if it doesn't significantly reduce the waveform size.  If false, the function will only compress the waveform if a size reduction is achieved. Valid values: true or false | `true` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `s` | struct | A structure containing the compressed waveform data.  It has two fields:
- `num_samples`: The number of samples in the original uncompressed waveform `w`.
- `data`: The compressed waveform data. |

## Examples

```matlab
s = mr.compressShape_mat([1, 1, 1, 2, 2, 3, 3, 3, 2, 2, 1, 1]);
s = mr.compressShape_mat(myWaveform, true);
```

## See Also

[decompressShape](decompressShape.md)
