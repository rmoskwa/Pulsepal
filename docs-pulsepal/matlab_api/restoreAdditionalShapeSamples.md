# restoreAdditionalShapeSamples

This function post-processes a gradient waveform, specifically addressing issues that can arise when trapezoidal gradients are converted into arbitrary shapes.  It aims to restore samples at the edges of gradient raster intervals to ensure accuracy.  The function identifies and corrects discrepancies between the reconstructed waveform and the original, particularly important in situations like spiral gradients where small deviations can occur.

## Syntax

```matlab
function [tt_chg, waveform_chg] = restoreAdditionalShapeSamples(tt,waveform,first,last,gradRasterTime,iBlock)
```

## Calling Pattern

```matlab
mr.restoreAdditionalShapeSamples(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `tt` | double | Time vector corresponding to the original gradient waveform. | `[0 0.001 0.002 0.003]` | seconds |
| `waveform` | double | Original gradient waveform amplitude values. | `[0 100 100 0]` | Hz/m |
| `first` | double | Amplitude of the first sample of the original gradient waveform. | `0` | Hz/m |
| `last` | double | Amplitude of the last sample of the original gradient waveform. | `0` | Hz/m |
| `gradRasterTime` | double | Time interval of the gradient raster. | `0.000001` | seconds |
| `iBlock` | double | Index of the current block (optional, used for warning messages). | `1` |  |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `tt_chg` | double | Modified time vector after shape restoration. |
| `waveform_chg` | double | Modified gradient waveform after shape restoration. |

## Examples

```matlab
% Basic shape restoration
tt = [0, 1e-3, 2e-3, 3e-3];
waveform = [0, 100, 100, 0];
[tt_chg, waveform_chg] = mr.restoreAdditionalShapeSamples(tt, waveform, 0, 0, 1e-6, 1);

% Restore gradient shape samples
[tt_restored, waveform_restored] = mr.restoreAdditionalShapeSamples(grad.tt, grad.waveform, ...
                                                                   grad.first, grad.last, ...
                                                                   gradRasterTime, blockIndex);

% Process arbitrary gradient waveform
if ~isempty(grad.tt)
    [tt_chg, waveform_chg] = mr.restoreAdditionalShapeSamples(grad.tt, grad.waveform, ...
                                                             grad.first, grad.last, ...
                                                             obj.gradRasterTime, iBc);
end
```
