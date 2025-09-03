# calcAdcSeg

This function calculates the optimal splitting of an ADC (Analog-to-Digital Converter) signal into segments for compatibility with specific MRI scanner hardware, particularly Siemens scanners.  Scanners may have limitations on the maximum number of samples in a single ADC object. This function determines the number of segments and samples per segment, ensuring that each segment's length adheres to the scanner's constraints while minimizing the total number of segments.  The function considers the gradient raster time and dwell time to align segments with the gradient waveform.  It offers two modes: 'shorten' (reducing the total number of samples if necessary) and 'lengthen' (increasing the number of samples to satisfy constraints).

## Syntax

```matlab
function [adcSegments,adcSamplesPerSegment] = calcAdcSeg(numSamples,dwell,system,mode)
```

## Calling Pattern

```matlab
mr.calcAdcSeg(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `numSamples` | double | The total number of ADC samples. | `16384` |  |
| `dwell` | double | The dwell time (sampling interval) in seconds. | `4e-6` | seconds |
| `system` | struct | A structure containing system parameters.  It must include fields like 'adcSamplesLimit' (maximum number of samples per ADC segment), 'adcSamplesDivisor' (divisor for the number of samples per segment), 'gradRasterTime' (gradient raster time), and 'adcRasterTime' (ADC raster time). | `mr.opts()` |  |
| `mode` | char | Specifies how to handle the number of samples if the initial configuration does not satisfy constraints.  'shorten' reduces the number of samples to meet constraints; 'lengthen' increases the number of samples. | `'shorten'` |  |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `adcSegments` | double | The optimal number of ADC segments. |
| `adcSamplesPerSegment` | double | The number of samples per ADC segment. |

## Examples

```matlab
[adcSegments, adcSamplesPerSegment] = mr.calcAdcSeg(16384, 4e-6, mr.opts(), 'shorten')
```

## See Also

[mr.opts](opts.md)
