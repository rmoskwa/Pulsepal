# getSupportedRfUse

This function returns a cell array containing strings representing the supported RF pulse uses in Pulseq, and optionally a corresponding array of single-character abbreviations.

## Syntax

```matlab
function [supported_rf_use, short_rf_use] = getSupportedRfUse()
```

## Calling Pattern

```matlab
mr.getSupportedRfUse(...)
```

## Parameters

*No parameters*

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `supported_rf_use` | cell | A cell array of strings representing the supported RF pulse uses.  The strings are: 'excitation', 'refocusing', 'inversion', 'saturation', 'preparation', 'other', 'undefined'. |
| `short_rf_use` | char | An array of single characters representing abbreviated forms of the supported RF pulse uses. This is only returned if a second output argument is requested.  It contains the first letter of each string in `supported_rf_use`. |

## Examples

```matlab
[supportedUses] = mr.getSupportedRfUse();
[supportedUses, shortUses] = mr.getSupportedRfUse();
```
