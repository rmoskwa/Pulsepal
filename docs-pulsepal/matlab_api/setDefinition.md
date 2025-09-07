# setDefinition

This function modifies or creates a custom definition within a Pulseq sequence object.  It allows users to store key-value pairs as metadata associated with the sequence.

## Syntax

```matlab
function setDefinition(key,val)
```

## Calling Pattern

```matlab
seq.setDefinition(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `key` | char | The name of the definition to set or modify. This acts as the key for the key-value pair. | `'FOV'` |  |
| `val` | double|string|cell | The value to assign to the specified definition key. The type of value depends on the definition. For example, it could be a numerical value for field of view (FOV), or a string. | `[0.2, 0.2, 0.2]` | varies |

## Examples

```matlab
% Set field of view for 3D imaging sequence
seq.setDefinition('FOV', [fov fov sliceThickness]);

% Define sequence name for scanner identification
seq.setDefinition('Name', 'gre');

% Configure readout oversampling for EPI sequences
seq.setDefinition('ReadoutOversamplingFactor', ro_os);
```

## See Also

[getDefinition](getDefinition.md)
