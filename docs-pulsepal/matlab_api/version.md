# version

Returns the version information for the Pulseq Matlab package.  The specific version information returned ('pulseq' or 'output') depends on the input parameter `type`.

## Syntax

```matlab
function [version_major, version_minor, version_revision, version_combined]=version(type)
```

## Calling Pattern

```matlab
mr.aux.version(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `type` | char | Specifies the type of version information to return.  'pulseq' returns the version of the Matlab package, while 'output' returns the version number written to the output file by the `seq.write()` function. | `'pulseq'` |  |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `version_major` | double | Major version number. |
| `version_minor` | double | Minor version number. |
| `version_revision` | double | Revision version number. |
| `version_combined` | double | Combined version number (major*1000000 + minor*1000 + revision). |

## Examples

```matlab
mr.aux.version('pulseq')
mr.aux.version('output')
```

## See Also

[seq.write](write.md)
