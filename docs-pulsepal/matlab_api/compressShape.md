# compressShape

Compresses a gradient or pulse waveform using a run-length encoding scheme applied to its derivative.  This efficiently represents waveforms with constant or linearly changing segments using significantly fewer samples. The function returns a structure containing the compressed waveform and the original number of samples.

## Syntax

```matlab
function s=compressShape(w, forceCompression)
```

## Calling Pattern

```matlab
mr.compressShape(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `w` | double | The input waveform (gradient or pulse shape) to be compressed.  This is a vector of waveform amplitude values. | `[1, 1, 1, 2, 2, 3, 3, 3, 3]` |  |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `forceCompression` | logical | `false` | A flag indicating whether to force compression even if the input waveform is short. If false, waveforms with 4 or fewer samples are returned uncompressed. Valid values: true, false | `true` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `s` | struct | A structure containing the compressed waveform data.  The structure has two fields:
- `num_samples`: The number of samples in the original, uncompressed waveform.
- `data`: A vector containing the compressed waveform data in a format suitable for Pulseq. |

## Examples

```matlab
s = mr.compressShape([1 1 1 2 2 3 3 3 3]);
% Compress a sample waveform
s = mr.compressShape([1 2 3 4], true);
```

## See Also

[decompressShape](decompressShape.md)
