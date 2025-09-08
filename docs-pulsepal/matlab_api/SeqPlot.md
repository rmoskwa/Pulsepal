# SeqPlot

This function generates a plot visualizing different aspects of a Pulseq sequence.  It displays the ADC/labels, RF magnitude, RF/ADC phase, and gradient waveforms (Gx, Gy, Gz) over time.  The plot can be customized using various optional parameters to control the time range, block range, time units, labels to display, whether to stack plots, show blocks, and display guides.

## Syntax

```matlab
function obj = SeqPlot(seq, varargin)
```

## Calling Pattern

```matlab
seqplot = mr.aux.SeqPlot(...)
seqplot = mr.aux.SeqPlot('ParameterName', value, ...)
```

## Parameters

This function accepts both positional parameters and name-value pairs.

### Name-Value Pair Arguments
| Parameter Name (string) | Value Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `'showBlocks'` | logical | `false` | Specifies whether to show block boundaries on the plot.  Accepts logical true/false or numeric 1/0. Valid values: true/false or 1/0 | `true` |
| `'timeRange'` | double | `[0 inf]` | Defines the time range to display on the plot.  Must be a 2-element vector [start_time end_time]. Valid values: 2-element numeric vector (Units: seconds) | `[0 0.1]` |
| `'blockRange'` | double | `[1 inf]` | Defines the range of blocks from the sequence to include in the plot. Must be a 2-element vector [start_block end_block]. Valid values: 2-element numeric vector | `[1 10]` |
| `'timeDisp'` | char | `validTimeUnits{1}` | Specifies the units for the time axis of the plot. Valid values: 's', 'ms', 'us' | `'ms'` |
| `'label'` | char | `[]` | Specifies which labels to display on the plot.  Accepts a string or array of strings. Valid values: mr.getSupportedLabels() | `'excitation'` |
| `'hide'` | logical | `false` | Specifies whether to hide the generated figure. Accepts logical true/false or numeric 1/0. Valid values: true/false or 1/0 | `false` |
| `'stacked'` | logical | `false` | Specifies whether to stack the plots vertically. Only works in MATLAB, not Octave. Valid values: true/false or 1/0 | `true` |
| `'showGuides'` | logical | `true` | Specifies whether to show guides on the plot. Valid values: true/false or 1/0 | `true` |

## Returns

| Output | Value Type | Description |
|--------|------|-------------|
| `obj` | struct | A structure containing handles to the generated figure and axes. |

## Examples

```matlab
seqplot = seq.mr.aux.SeqPlot('timeRange', [0, 0.05], 'timeDisp', 'ms', 'label', 'excitation')
```

## See Also

[mr.getSupportedLabels](getSupportedLabels.md)
