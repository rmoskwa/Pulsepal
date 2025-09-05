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
value = seq.getDefinition('FOV');
myValue = seq.getDefinition('myCustomParam');
```

## See Also

[setDefinition](setDefinition.md)
