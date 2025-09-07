# getDefinition

Retrieves the value associated with a specified key from a custom definitions structure within a Pulseq sequence object.  This function allows access to user-defined parameters stored in the sequence object.

## Syntax

```matlab
function value=getDefinition(key)
```

## Calling Pattern

```matlab
seq.getDefinition(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `key` | char | The key string identifying the desired definition. | `'FOV'` |  |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `trajectory_delay` | double | `0` | This parameter is not used in the provided code snippet. Valid values: any non-negative number (Units: seconds) | `0.001` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `value` | any | The value associated with the specified key. Returns an empty array if the key is not found. |

## Examples

```matlab
% Retrieve field of view from sequence definition
fov = seq.getDefinition('FOV');

% Check sequence name for conditional reconstruction
if strcmp('petra',seq.getDefinition('Name'))
    SamplesPerShell = seq.getDefinition('SamplesPerShell');
end

% Get sequence name for processing pipeline decisions
seqName = seq.getDefinition('Name');
```

## See Also

[setDefinition](setDefinition.md)
