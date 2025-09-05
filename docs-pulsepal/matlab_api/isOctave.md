# isOctave

This function checks if the code is running within the Octave environment. It uses a persistent variable to store the result of the check, avoiding repeated calls to the `exist` function.

## Syntax

```matlab
function OUT = isOctave ()
```

## Calling Pattern

```matlab
mr.aux.isOctave(...)
```

## Parameters

*No parameters*

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `OUT` | double | A logical value (1 or 0) indicating whether the code is running in Octave (1 for Octave, 0 for MATLAB). |

## Examples

```matlab
mr.aux.isOctave()
```
