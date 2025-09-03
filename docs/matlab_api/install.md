# install

Installs a Pulseq sequence directly on an MRI scanner system. This function facilitates deployment of sequences to scanner platforms, particularly Siemens systems, by copying the sequence files to the appropriate scanner directories and optionally to the RANGE controller.

## Syntax

```matlab
function ok=install(param1,param2)
```

## Calling Pattern

```matlab
seq.install(...)
```

## Parameters

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `param1` | char | | Scanner type or sequence path/name. If omitted, auto-detects scanner environment. Options: 'siemens' (Numaris4), 'siemensNX' (NumarisX), or a custom sequence path/name | `'siemens'` or `'my_seq/custom_name'` |
| `param2` | char | | Sequence path or name when param1 specifies scanner type. Creates subdirectories automatically if provided | `'sequences/my_epi'` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `ok` | logical | Returns true if installation was successful, false otherwise |

## Usage Modes

The function supports several usage patterns:

1. **Auto-detect scanner**: `seq.install()` - Automatically detects the scanner environment and installs with default name
2. **Custom name with auto-detect**: `seq.install('sequence_path_or_name')` - Auto-detects scanner and installs with specified name
3. **Siemens Numaris4**: `seq.install('siemens')` - Installs as external.seq on Numaris4 system
4. **Siemens NumarisX**: `seq.install('siemensNX')` - Installs as external.seq on NumarisX system
5. **Siemens with custom name**: `seq.install('siemens', 'my_seq/custom_name')` - Installs with custom path/name on Numaris4
6. **NumarisX with custom name**: `seq.install('siemensNX', 'my_seq/custom_name')` - Installs with custom path/name on NumarisX

## Examples

```matlab
% Auto-detect scanner and install with default name
seq.install();

% Install with custom name (auto-detect scanner)
seq.install('protocols/my_epi_sequence');

% Install on Siemens Numaris4 system as external.seq
seq.install('siemens');

% Install on Siemens NumarisX with custom name
seq.install('siemensNX', 'research/diffusion_seq');

% Install with subdirectory creation
seq.install('siemens', 'project_2024/sequences/gre_v2');
```

## Notes

- The function uses network ping to verify scanner connectivity before installation
- Subdirectories specified in the path are created automatically
- The function is typically commented out during sequence development and uncommented for deployment
- Installation requires appropriate network access and permissions to the scanner system
- On Windows systems, uses Windows-specific ping parameters; on Unix/Mac, uses Unix-style ping

## See Also

[write](write.md), [read](read.md), [checkTiming](checkTiming.md)
