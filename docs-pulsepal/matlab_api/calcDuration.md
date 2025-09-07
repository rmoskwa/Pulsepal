# calcDuration

Calculates the duration of a Pulseq event, a sequence of events, or a block structure.  It iterates through the events (delays, RF pulses, gradients, ADCs, traps, outputs, and triggers), determining the maximum duration among them.  The function handles both individual events and block structures, converting the latter into a cell array of events before processing.

## Syntax

```matlab
function duration=calcDuration(varargin)
```

## Calling Pattern

```matlab
mr.calcDuration(...)
```

## Parameters


### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `delay` | struct | `N/A` | A Pulseq delay event structure. Contains at least a 'delay' field specifying the delay duration in seconds. (Units: seconds) | `{type: 'delay', delay: 0.001}` |
| `rf` | struct | `N/A` | A Pulseq RF pulse event structure.  Contains 'delay', 'shape_dur' (pulse duration), and 'ringdownTime' fields specifying durations in seconds. (Units: seconds) | `{type: 'rf', delay: 0.0001, shape_dur: 0.0005, ringdownTime: 0.0001}` |
| `grad` | struct | `N/A` | A Pulseq gradient event structure.  Contains 'delay' and 'shape_dur' fields specifying durations in seconds. (Units: seconds) | `{type: 'grad', delay: 0.0002, shape_dur: 0.001}` |
| `adc` | struct | `N/A` | A Pulseq ADC event structure. Contains 'delay', 'numSamples', 'dwell', and 'deadTime' fields.  'delay' and 'deadTime' are in seconds, 'dwell' is the sampling time in seconds. (Units: seconds) | `{type: 'adc', delay: 0.0001, numSamples: 128, dwell: 0.000001, deadTime: 0.00005}` |
| `trap` | struct | `N/A` | A Pulseq trapezoidal gradient event structure. Contains 'delay', 'riseTime', 'flatTime', and 'fallTime' fields specifying durations in seconds. (Units: seconds) | `{type: 'trap', delay: 0.0001, riseTime: 0.0002, flatTime: 0.001, fallTime: 0.0002}` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `duration` | double | The maximum duration (in seconds) among all provided events or within the block structure. |

## Examples

```matlab
% Calculate TE delay accounting for gradient and RF timings
delayTE = ceil((TE - mr.calcDuration(gxPre) - gz.fallTime - gz.flatTime/2 ...
    - mr.calcDuration(gx)/2)/seq.gradRasterTime)*seq.gradRasterTime;

% Calculate TR delay accounting for all sequence events
delayTR = ceil((TR - mr.calcDuration(gz) - mr.calcDuration(gxPre) ...
    - mr.calcDuration(gx) - delayTE)/seq.gradRasterTime)*seq.gradRasterTime;

% Create spoiler gradient with duration based on RF pulse
gz_fs = mr.makeTrapezoid('z', sys, 'delay', mr.calcDuration(rf_fs), 'Area', 1/1e-4);
```
