# write_v141

Writes a Pulseq sequence object to a file in version 1.4.1 format.  This function takes a Pulseq sequence object and a filename as input and writes the sequence data to the specified file using the Pulseq open file format. It also includes an optional parameter to control whether a signature is created.

## Syntax

```matlab
function write_v141(filename,create_signature)
```

## Calling Pattern

```matlab
seq.write_v141(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `filename` | char | The name of the file to which the sequence data should be written.  This includes the full path. | `'my_sequences/gre.seq'` |  |
| `create_signature` | logical | A boolean flag indicating whether a signature should be created for the file (default is true if not provided). | `true` |  |

## Examples

```matlab
seq.write_v141('my_sequences/gre.seq', true);
seq.write_v141('my_sequences/gre.seq');
```

## See Also

[mr.aux.version](version.md), [write](write.md), [read](read.md)
