# applyToBlock

This helper function processes a sequence of Pulseq events (potentially within a block) and converts it into a standardized cell array.  It separates the events into different categories: RF pulses, ADC events, gradient events (x, y, z), and other miscellaneous events. It also handles the case where the input is a single structure representing a block of events or a cell array containing such structures.  The function updates the object's label settings based on 'NOPOS', 'NOROT', and 'NOSCL' labels present in the input events.

## Syntax

```matlab
function out=applyToBlock(varargin)
```

## Calling Pattern

```matlab
tra.applyToBlock(...)
```

## Parameters


### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `rf` | struct |  | RF pulse event structure to transform | `rf_event` |
| `grad` | struct |  | Gradient event structure to transform | `grad_event` |
| `adc` | struct |  | ADC event structure to transform | `adc_event` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `out` | cell | A cell array containing the processed Pulseq events. The structure of this cell array is not explicitly defined but likely follows a format consistent with Pulseq's internal representation of sequence events. The exact structure depends on the content of the input 'varargin'. |

## Examples

```matlab
out = tra.applyToBlock(rf_event, grad_event, adc_event);
```
