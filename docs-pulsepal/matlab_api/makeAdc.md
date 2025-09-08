# makeAdc

Creates an ADC readout event for Pulseq sequence design.  It defines parameters for the analog-to-digital conversion process, including the number of samples, dwell time, duration, delay, and frequency and phase offsets.  It can account for system-specific dead times.

## Syntax

```matlab
function adc=makeAdc(num,varargin)
```

## Calling Pattern

```matlab
mr.makeAdc(...)
mr.makeAdc('ParameterName', value, ...)
```

## Parameters

This function accepts both positional parameters and name-value pairs.

### Required Parameters

| Parameter Name | Value Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `num` | double | The number of samples to be acquired by the ADC. | `1024` |  |

### Name-Value Pair Arguments
| Parameter Name (string) | Value Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `system` | struct | `[]` | A structure containing system parameters.  If not provided, default system parameters are used.  Should contain the field `adcDeadTime` (in seconds). | `mr.opts()` |
| `dwell` | double | `0` | The dwell time (sampling interval) in seconds.  Must be specified if `duration` is not. Valid values: dwell > 0 (Units: seconds) | `0.000004` |
| `duration` | double | `0` | The total duration of the ADC readout in seconds. Must be specified if `dwell` is not. Valid values: duration > 0 (Units: seconds) | `0.004` |
| `delay` | double | `0` | The delay before the ADC readout begins in seconds. (Units: seconds) | `0.001` |
| `freqOffset` | double | `0` | Frequency offset of the ADC readout in Hz. (Units: Hz) | `100` |
| `phaseOffset` | double | `0` | Phase offset of the ADC readout in radians. (Units: radians) | `pi/2` |
| `freqPPM` | double | `0` | Frequency offset in parts per million (ppm). (Units: ppm) | `10` |
| `phasePPM` | double | `0` | Phase offset in parts per million (ppm). (Units: ppm) | `5` |
| `phaseModulation` | double | `[]` | A vector of phase modulation values for each sample in radians. Must be the same length as `num`. (Units: radians) | `[0, pi/4, pi/2, 3*pi/4, pi]` |

## Returns

| Output | Value Type | Description |
|--------|------|-------------|
| `adc` | struct | A structure containing the ADC readout parameters.  Fields include: `type`, `numSamples`, `dwell`, `delay`, `freqOffset`, `phaseOffset`, `freqPPM`, `phasePPM`, `deadTime`, `duration`, and `phaseModulation`. |

## Examples

```matlab
% Basic ADC with specified duration
adc = mr.makeAdc(512, 'Duration', 512e-3, 'system', system, 'delay', system.adcDeadTime);

% ADC synchronized with readout gradient
gx = mr.makeTrapezoid('x', sys, 'FlatArea', Nx*deltak, 'FlatTime', roDuration);
adc = mr.makeAdc(Nx, sys, 'Duration', gx.flatTime, 'Delay', gx.riseTime);

% ADC with oversampling
ro_os = 2; % oversampling factor
adc = mr.makeAdc(Nx*ro_os, 'Duration', ro_dur, 'Delay', gro.riseTime, 'system', sys);
```

## See Also

[Sequence.addBlock](addBlock.md)
