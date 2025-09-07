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
% Write sequence to binary format for testing
seq.writeBinary('zte_petra.bin');

% Read back binary sequence and verify
seq_bin = mr.Sequence();
seq_bin.readBinary('zte_petra.bin');
seq_bin.write('zte_petra_bin.seq');

% Basic binary file writing
seq.writeBinary('sequences/gre.bin');
```

## See Also

[readBinary](readBinary.md)
