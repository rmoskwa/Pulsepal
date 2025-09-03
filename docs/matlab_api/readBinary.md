# readBinary

Loads a Pulseq sequence from a binary file.  This function reads a binary file containing a Pulseq sequence and populates a Pulseq sequence object with the data.  The binary file format is described in the Pulseq specification at http://pulseq.github.io.  The function reads various sections of the binary file, including definitions, blocks, RF pulses, gradients, ADC events, delays, and shapes, and stores them in the appropriate fields of the sequence object.

## Syntax

```matlab
function readBinary(filename)
```

## Calling Pattern

```matlab
seq.readBinary(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `filename` | string | The path to the binary file containing the Pulseq sequence. | `'sequences/gre.bin'` |  |

## Examples

```matlab
seq.readBinary('sequences/gre.bin')
```

## See Also

[writeBinary](writeBinary.md)
