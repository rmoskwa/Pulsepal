# waveforms_and_times

This function extracts and organizes waveform data from a Pulseq sequence object. It decompresses the gradient waveforms, returning them as a cell array where each cell contains time points and corresponding gradient amplitude values for each gradient axis (typically x, y, and z).  It also returns timing information for excitation and refocusing RF pulses, and ADC sampling points, including frequency and phase offsets.  If the `appendRF` flag is true, RF waveforms are included in the output.

## Syntax

```matlab
function [wave_data, tfp_excitation, tfp_refocusing, t_adc, fp_adc, pm_adc]=waveforms_and_times(appendRF, blockRange)
```

## Calling Pattern

```matlab
seq.waveforms_and_times(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `appendRF` | logical | A boolean flag indicating whether to include RF waveforms in the output.  True includes RF data; False excludes it. | `true` |  |
| `blockRange` | numeric | A two-element vector specifying the range of blocks in the sequence to process.  The first element is the starting block index, and the second element is the ending block index. | `[1, 10]` |  |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `trajectory_delay` | double | `0` | Delay added to the trajectory. (Units: seconds) | `0.001` |
| `gradient_offset` | double | `0` | Offset added to the gradient waveforms. (Units: Hz/m) | `100` |
| `blockRange` | numeric | `[1 inf]` | A two-element vector specifying the range of blocks in the sequence to process. Defaults to processing all blocks. | `[5,15]` |
| `externalWaveformsAndTimes` | struct | `struct([])` | Allows for providing external waveforms and timing data. | `struct('gradients', { [1 2], [3 4] }, 'rf', { [5 6], [7 8] })` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `wave_data` | cell | A cell array containing the gradient waveforms. Each cell represents a gradient axis (x, y, z) and contains a matrix of time points and corresponding amplitude values. |
| `tfp_excitation` | numeric | Time points, frequency, and phase offsets of the excitation RF pulses. |
| `tfp_refocusing` | numeric | Time points, frequency, and phase offsets of the refocusing RF pulses. |
| `t_adc` | numeric | Time points of all ADC sampling points. |
| `fp_adc` | numeric | Frequency and phase offsets of each ADC object. |
| `pm_adc` | numeric | Phase modulation of every ADC sample beyond the data stored in fp_adc. |

## Examples

```matlab
[wave_data, tfp_excitation, tfp_refocusing, t_adc, fp_adc, pm_adc] = seq.waveforms_and_times(true, [1, 10]);
```

## See Also

[getBlock](getBlock.md)
