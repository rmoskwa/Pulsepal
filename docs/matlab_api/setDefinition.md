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
seq.setDefinition('FOV', [0.2, 0.2, 0.2]); % Sets the FOV definition
seq.setDefinition('PatientName', 'John Doe'); % Sets the patient name definition
```

## See Also

[getDefinition](getDefinition.md)
