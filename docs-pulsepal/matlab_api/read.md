# read

Loads a Pulseq sequence from a specified file into a sequence object.  The function reads definitions, signature information, and sequence data from the file, populating the object's properties accordingly.  It handles optional parameters to detect RF pulse usage and provides error handling for file opening.

## Syntax

```matlab
function read(filename,varargin)
```

## Calling Pattern

```matlab
seq.read(...)
```

## Parameters


### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `filename` | char | `N/A` | The name of the file containing the Pulseq sequence data. | `'my_sequences/gre.seq'` |
| `major` | double | `N/A` | Not explicitly used in provided code, likely part of version information (unused). Valid values: N/A | `1` |
| `minor` | double | `N/A` | Not explicitly used in provided code, likely part of version information (unused). Valid values: N/A | `0` |
| `revision` | double | `N/A` | Not explicitly used in provided code, likely part of version information (unused). Valid values: N/A | `0` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `none` | none | The function modifies the input `obj` directly.  It does not return a value. |

## Examples

```matlab
% Load sequence for reconstruction processing
seq.read(seq_file_path,'detectRFuse');

% Simple sequence loading
seq2.read(seq_name);

% Load sequence with RF use detection for analysis
seq.read(seq_file_path,'detectRFuse');
```

## See Also

[write](write.md)
