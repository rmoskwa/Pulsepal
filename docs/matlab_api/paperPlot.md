# paperPlot

This helper function generates a plot of a Pulseq sequence in a style suitable for scientific publications. It allows customization of various aspects of the plot's appearance, such as line width, axes color, and the colors of different gradient and RF components.

## Syntax

```matlab
function sp = paperPlot(varargin)
```

## Calling Pattern

```matlab
seq.paperPlot(...)
```

## Parameters


### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `blockRange` | numeric array | `[1 inf]` | Specifies the range of blocks in the sequence to be plotted.  It's a two-element vector [first last], where 'first' and 'last' are the indices of the first and last blocks to include, respectively. Defaults to plotting the entire sequence. Valid values: Two-element numeric array | `[5 10]` |
| `lineWidth` | double | `1.2` | Specifies the width of the lines used in the plot. Valid values: Positive numeric value | `2.0` |
| `axesColor` | char or 1x3 numeric array | `[0.5 0.5 0.5]` | Specifies the color of the horizontal axes. Can be a standard color name (e.g., 'red', 'blue'), a hexadecimal RGB string (e.g., '#FF0000'), or a 1x3 vector of RGB values (e.g., [1 0 0]). Valid values: Valid MATLAB color specification | `'black'` |
| `rfColor` | char or 1x3 numeric array | `'black'` | Specifies the color of the RF and ADC events in the plot.  Uses the same color specification as axesColor. Valid values: Valid MATLAB color specification | `[0 1 0]` |
| `gxColor` | char or 1x3 numeric array | `'blue'` | Specifies the color of the X gradients in the plot. Uses the same color specification as axesColor. Valid values: Valid MATLAB color specification | `'red'` |
| `gyColor` | char or 1x3 numeric array | `'red'` | Specifies the color of the Y gradients in the plot. Uses the same color specification as axesColor. Valid values: Valid MATLAB color specification | `'green'` |
| `gzColor` | char or 1x3 numeric array | `[0 0.5 0.3]` | Specifies the color of the Z gradients in the plot. Uses the same color specification as axesColor. Valid values: Valid MATLAB color specification | `'cyan'` |
| `rfPlot` | char | `'abs'` | Specifies how RF pulses are plotted: 'abs' for the magnitude, 'real' for the real part, or 'imag' for the imaginary part. Valid values: 'abs', 'real', 'imag' | `'real'` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `sp` | double | Handle to the generated figure. |

## Examples

```matlab
sp = seq.paperPlot();
sp = seq.paperPlot('blockRange', [10 20], 'lineWidth', 2, 'rfColor', 'red');
```
