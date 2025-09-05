# pts2waveform

This function interpolates a set of points (times, amplitudes) to generate a waveform with a specified gradient raster time.  It takes time points and corresponding amplitude values as input, and outputs a waveform sampled at the specified raster time. The interpolation ensures a smooth waveform consistent with the gradient hardware's resolution.

## Syntax

```matlab
function waveform = pts2waveform(times, amplitudes, gradRasterTime)
```

## Calling Pattern

```matlab
mr.pts2waveform(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `times` | double | A vector of time points at which the amplitudes are defined. | `[0, 0.001, 0.002]` | seconds |
| `amplitudes` | double | A vector of amplitude values corresponding to the time points in 'times'. | `[0, 100, 0]` | Hz/m |
| `gradRasterTime` | double | The time resolution of the gradient waveform. The output waveform will be sampled at this interval. | `0.0001` | seconds |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `waveform` | double | A vector representing the interpolated gradient waveform, sampled at the specified 'gradRasterTime'. |

## Examples

```matlab
waveform = mr.pts2waveform([0, 0.001, 0.002], [0, 100, 0], 0.0001);
```
