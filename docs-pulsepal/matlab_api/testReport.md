# testReport

Analyzes a Pulseq sequence object and generates a text report summarizing key sequence parameters.  The report includes information such as RF flip angles, echo time (TE), repetition time (TR), and other relevant sequence characteristics.  An optional 'system' parameter allows for comparison of sequence parameters against the limits of a specified MR system.

## Syntax

```matlab
function [ report ] = testReport(varargin )
```

## Calling Pattern

```matlab
seq.testReport(...)
seq.testReport('ParameterName', value, ...)
```

## Parameters

### Name-Value Pair Arguments
| Parameter Name (string) | Value Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `system` | struct | `struct([])` | A structure containing MR system specifications to compare against. If omitted, no system-specific checks are performed. Valid values: A structure with fields representing system limits (e.g., gradient limits, slew rate limits, etc.). The exact fields depend on the specific tests performed by the function. | `{ 'maxGrad': 40, 'maxSlew': 120 }` |

## Returns

| Output | Value Type | Description |
|--------|------|-------------|
| `report` | string | A text string containing the analysis report of the input Pulseq sequence. |

## Examples

```matlab
% Generate sequence analysis report
rep = seq.testReport;
fprintf([rep{:}]);

% Simple test report generation
rep = seq.testReport();

% Generate report for development testing
rep = seq.testReport;
disp(rep);
```
