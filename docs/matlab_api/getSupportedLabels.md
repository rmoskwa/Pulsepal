# getSupportedLabels

This function returns a cell array of strings representing supported labels for Pulseq sequence parameters.  These labels are used to map sequence parameters to specific fields in the scanner's data handling system (MDH), particularly for Siemens scanners.  The labels cover various aspects of pulse sequence design, including repetition, averaging, phase cycling, parallel imaging, motion correction, and execution control.

## Syntax

```matlab
function supported_labels = getSupportedLabels()
```

## Calling Pattern

```matlab
mr.getSupportedLabels(...)
```

## Parameters

*No parameters*

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `supported_labels` | cell | A cell array of strings containing the supported labels.  Each string represents a specific label used in Pulseq sequence design and mapping to scanner parameters. |

## Examples

```matlab
supportedLabels = mr.getSupportedLabels();
```
