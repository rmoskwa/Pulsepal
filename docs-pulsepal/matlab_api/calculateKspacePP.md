# calculateKspacePP

Calculates the k-space trajectory of a Pulseq sequence using a piecewise-polynomial gradient wave representation. This method is efficient for sequences with simple gradient shapes and long delays.  It considers optional parameters for compensating ADC and gradient timing mismatches and simulating background gradients.

## Syntax

```matlab
function [ktraj_adc, t_adc, ktraj, t_ktraj, t_excitation, t_refocusing, slicepos, t_slicepos, gw_pp, pm_adc] = calculateKspacePP(varargin)
```

## Calling Pattern

```matlab
seq.calculateKspacePP(...)
seq.calculateKspacePP('ParameterName', value, ...)
```

## Parameters

This function accepts both positional parameters and name-value pairs.

### Name-Value Pair Arguments
| Parameter Name (string) | Value Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `'trajectory_delay'` | double | `0` | A compensation factor to align the ADC and gradient timings in k-space reconstruction.  Positive values delay the gradient relative to the ADC. Valid values: Any numeric value (Units: seconds) | `0.000001` |
| `'gradient_offset'` | double | `0` | Simulates background gradients or helps verify spin-echo conditions by adding a constant offset to the gradient waveforms. Valid values: Any numeric value (Units: Hz/m) | `10` |
| `'blockRange'` | double | `[1 inf]` | Specifies a range of blocks within the sequence to consider for k-space trajectory calculation. Valid values: A two-element numeric array [start_block end_block], where end_block can be Inf to include all blocks from start_block to the end of the sequence.  start_block must be >=1 | `[5,10]` |
| `'externalWaveformsAndTimes'` | struct | `struct([])` | Allows supplying external gradient waveforms and timing information instead of using those from the Pulseq sequence object.  The struct should contain fields 'gw_data', 'tfp_excitation', 'tfp_refocusing', and 't_adc'.  It may optionally include 'pm_adc'. Valid values: A struct with fields: gw_data (gradient waveforms), tfp_excitation (excitation pulse timings), tfp_refocusing (refocusing pulse timings), t_adc (ADC timings), optionally pm_adc (phase modulation for ADC). | `{gw_data: myGradientData, tfp_excitation: myExcitationTimes, tfp_refocusing: myRefocusingTimes, t_adc: myADCtimes}` |

## Returns

| Output | Value Type | Description |
|--------|------|-------------|
| `ktraj_adc` | double | k-space trajectory corresponding to ADC sampling times. |
| `t_adc` | double | Time points corresponding to ADC samples. |
| `ktraj` | double | Complete k-space trajectory. |
| `t_ktraj` | double | Time points corresponding to the complete k-space trajectory. |
| `t_excitation` | double | Time points of excitation pulses. |
| `t_refocusing` | double | Time points of refocusing pulses. |
| `slicepos` | double | Slice position information. |
| `t_slicepos` | double | Time points corresponding to slice positions. |
| `gw_pp` | double | Piecewise polynomial representation of gradient waveforms (optional output). |
| `pm_adc` | double | Phase modulation for ADC (optional output). |

## Examples

```matlab
% Example 1: Basic k-space trajectory calculation with plotting
[ktraj_adc, t_adc, ktraj, t_ktraj, t_excitation, t_refocusing] = seq.calculateKspacePP();

% Plot k-space trajectory
figure; plot(t_ktraj, ktraj'); title('k-space components as functions of time');
figure; plot(ktraj(1,:), ktraj(2,:), 'b', ktraj_adc(1,:), ktraj_adc(2,:), 'r.');
axis('equal'); title('2D k-space');

% Example 2: Calculate trajectory with compensation for timing delays
[ktraj_adc, t_adc, ktraj, t_ktraj, t_excitation, t_refocusing] = seq.calculateKspacePP('trajectory_delay', [0 0 0]*1e-6);

% Example 3: Calculate trajectory for 3D visualization
[kfa, ta, kf] = seq.calculateKspacePP();
figure; plot3(kf(1,:), kf(2,:), kf(3,:));
hold on; plot3(kfa(1,:), kfa(2,:), kfa(3,:), 'r.');
```

## See Also

[waveforms_and_times](waveforms_and_times.md)
