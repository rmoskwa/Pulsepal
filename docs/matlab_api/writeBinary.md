# writeBinary

Writes a Pulseq sequence object to a file in binary format, using the Pulseq open file format specification.  This function serializes the sequence data, including header information, definitions, block events, RF pulses, and gradients, into a binary file for later use or sharing.

## Syntax

```matlab
function writeBinary(filename)
```

## Calling Pattern

```matlab
seq.writeBinary(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `filename` | char | The name of the file to write the sequence data to.  The filename should include the '.bin' extension. | `'sequences/gre.bin'` |  |

## Examples

```matlab
seq.writeBinary('sequences/gre.bin')
```

## See Also

[readBinary](readBinary.md)
