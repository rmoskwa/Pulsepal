# write

Writes a Pulseq sequence object to a file in the Pulseq open file format.  This function serializes the sequence data, including definitions, block events, and RF events, into a text-based file that can be later read back into MATLAB using the `read` function.

## Syntax

```matlab
function write(filename,create_signature)
```

## Calling Pattern

```matlab
seq.write(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `filename` | char | The name of the file to write the sequence data to, including the full path.  The file extension '.seq' is usually used. | `'my_sequences/gre.seq'` |  |
| `create_signature` | logical | A boolean value indicating whether to create a signature (MD5 hash) for the sequence file. If true (default), a signature is generated and included in the file.  | `true` |  |

## Examples

```matlab
seq.write('my_sequences/gre.seq')
seq.write('my_sequences/gre.seq', false)
```

## See Also

[read](read.md)
