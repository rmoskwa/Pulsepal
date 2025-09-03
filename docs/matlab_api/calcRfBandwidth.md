# calcRfBandwidth

Calculates the bandwidth of an RF pulse using a Fast Fourier Transform (FFT).  It assumes a low-angle approximation. The function returns the bandwidth, center frequency, and optionally the spectrum and frequency axis of the RF pulse.  It handles frequency offsets and resamples the pulse to a specified resolution before performing the FFT.

## Syntax

```matlab
function [bw,fc,spectrum,f,rfs,t]=calcRfBandwidth(rf, cutoff, df, dt)
```

## Calling Pattern

```matlab
mr.calcRfBandwidth(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `rf` | struct | Structure containing the RF pulse information.  This struct must include fields 't' (time vector), 'signal' (complex amplitude of the RF pulse), 'freqOffset' (frequency offset in Hz), 'freqPPM' (frequency offset in ppm), and 'phaseOffset' (phase offset in radians).  It also needs 'center', which represents a central point of the RF pulse used in resampling. | `{t=[0:1e-6:1e-3]; signal=exp(-(t-0.5e-3).^2/2e-6); freqOffset=0; freqPPM=0; phaseOffset=0; center=0.5e-3;}` |  |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `cutoff` | double | `0.5` | Fraction of the maximum amplitude used to define the bandwidth.  The bandwidth is determined by the points where the spectrum falls to this fraction of its maximum value. Valid values: [0, 1] | `0.1` |
| `df` | double | `10` | Frequency resolution of the FFT (spectral resolution). (Units: Hz) | `1` |
| `dt` | double | `1e-6` | Time resolution of the resampled RF pulse. (Units: seconds) | `5e-7` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `bw` | double | Bandwidth of the RF pulse in Hz. |
| `fc` | double | Center frequency of the RF pulse in Hz. |
| `spectrum` | double | Complex spectrum of the RF pulse. |
| `f` | double | Frequency axis corresponding to the spectrum. |
| `rfs` | double | Resampled RF pulse signal. |
| `t` | double | Time axis corresponding to the resampled RF pulse. |

## Examples

```matlab
  [bw, fc, spectrum, f, rfs, t] = mr.calcRfBandwidth(rf_pulse_struct, 0.2, 5, 1e-6);
```

## See Also

[mr.aux.findFlank](findFlank.md)
