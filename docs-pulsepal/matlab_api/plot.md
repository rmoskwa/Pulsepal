# plot

This function plots a Pulseq sequence object in a new figure.  It offers various options to control the appearance and content of the plot, including specifying time ranges, block ranges, color schemes, and display units for time. The function can also produce stacked plots and include dynamic guides for verifying event alignment. It utilizes the mr.aux.SeqPlot class internally for the actual plotting functionality.

## Syntax

```matlab
function sp = plot(varargin)
```

## Calling Pattern

```matlab
seq.plot(...)
```

## Parameters


### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `blockRange` | numeric | `[1 inf]` | Specifies the range of blocks in the sequence to be plotted.  A two-element vector [first last] indicating the first and last block indices to include. Defaults to plotting all blocks. | `[5 10]` |
| `lineWidth` | double | `1.2` | Specifies the line width for the plotted waveforms. | `2.0` |
| `axesColor` | numeric | `[0.5 0.5 0.5]` | Sets the color of the horizontal axes. A three-element RGB vector. | `[0.8 0.2 0.2]` |
| `rfColor` | char | `'black'` | Specifies the color of the RF and ADC events. | `'red'` |
| `gxColor` | char | `'blue'` | Specifies the color of the X gradients. | `'green'` |
| `gyColor` | char | `'red'` | Specifies the color of the Y gradients. | `'magenta'` |
| `gzColor` | numeric | `[0 0.5 0.3]` | Specifies the color of the Z gradients. A three-element RGB vector. | `[0.5 0 0.8]` |
| `rfPlot` | char | `'abs'` | Specifies how RF pulses are plotted: 'abs' (absolute value), 'real' (real part), or 'imag' (imaginary part). Valid values: 'abs', 'real', 'imag' | `'real'` |
| `timeRange` | numeric | `[]` | Specifies the time range to plot. A two-element vector [start stop] defining the start and end times in seconds. (Units: seconds) | `[0.01 0.05]` |
| `timeDisp` | char | `[]` | Specifies the units for time display: 's', 'ms', or 'us'. Valid values: 's', 'ms', 'us' | `'ms'` |
| `label` | char | `[]` | Specifies which ADC event labels to plot. A comma-separated string of label names. | `'LIN,REP'` |
| `showBlocks` | logical|numeric | `0` | If true (or 1), plots grid and tick labels at block boundaries. | `1` |
| `stacked` | logical|numeric | `0` | If true (or 1), arranges plots vertically in a stacked layout sharing the same x-axis. | `true` |
| `showGuides` | logical|numeric | `0` | If true (or 1), displays dynamic hairline guides that follow the data cursor to aid in verifying event alignment. | `1` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `sp` | object | A handle to the figure created by the plot function; returned only if one output argument is requested. |

## Examples

```matlab
seq.plot()
seq.plot('timeRange', [0.01 0.05], 'showBlocks', 1)
f = seq.plot('stacked', true, 'label', 'LIN,REP')
```

## See Also

[mr.aux.SeqPlot](SeqPlot.md), [paperPlot](paperPlot.md)
