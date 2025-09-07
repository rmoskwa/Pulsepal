# calcRfCenter

Calculates the effective center time point of an RF pulse. For shaped pulses, it determines the time point corresponding to the peak RF amplitude.  For block pulses, it calculates the center of the pulse.  Zeropadding is included in the calculation, but the RF pulse's delay field is ignored.

## Syntax

```matlab
function [tc ic]=calcRfCenter(rf)
```

## Calling Pattern

```matlab
mr.calcRfCenter(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `rf` | struct | A structure representing the RF pulse.  It must contain at least the 'signal' field (containing the RF amplitude values) and the 't' field (containing the corresponding time points). It may also contain a 'center' field. | `{signal: [0, 0.5, 1, 0.5, 0], t: [0, 1e-6, 2e-6, 3e-6, 4e-6]}` |  |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `tc` | double | The time point (in seconds) of the calculated RF pulse center. |
| `ic` | double | The index of the calculated RF pulse center within the rf.signal array. |

## Examples

```matlab
% Example 1: Calculate RF center for phase compensation
rf.freqOffset = gz.amplitude*thickness*(s-1-(Nslices-1)/2);
rf.phaseOffset = -2*pi*rf.freqOffset*mr.calcRfCenter(rf); % compensate for the slice-offset induced phase

% Example 2: Calculate timing delays using RF center
rfCenterInclDelay = rf.delay + mr.calcRfCenter(rf);
rf180centerInclDelay = rf180.delay + mr.calcRfCenter(rf180);
delayTE1 = ceil((TE/2 - mr.calcDuration(rf,gz) + rfCenterInclDelay - rf180centerInclDelay)/lims.gradRasterTime)*lims.gradRasterTime;

% Example 3: Compensate for frequency offset induced phase
rf_fs.phaseOffset = -2*pi*rf_fs.freqOffset*mr.calcRfCenter(rf_fs); % compensate for the frequency-offset induced phase
```
