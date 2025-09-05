# makeTrigger

Creates a trigger event for synchronization with an external signal.  This function generates a structure defining a trigger event for use within a Pulseq sequence.  It specifies the trigger channel, delay before the trigger, and duration after the trigger.  The duration is constrained to be at least as long as the system's gradient raster time.

## Syntax

```matlab
function trig = makeTrigger(channel, varargin)
```

## Calling Pattern

```matlab
mr.makeTrigger(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `channel` | char | The channel to use for the trigger.  Valid values are 'physio1' and 'physio2' (Siemens-specific). | `'physio1'` |  |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `delay` | double | `0` | The delay (in seconds) before the trigger event starts. (Units: seconds) | `0.001` |
| `duration` | double | `0` | The duration (in seconds) of the trigger event.  If shorter than the system's gradient raster time, it is automatically adjusted to match the raster time. (Units: seconds) | `0.005` |
| `system` | struct | `[]` | A structure containing system parameters (e.g., from mr.opts()). If not provided, mr.opts() is used. | `mr.opts()` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `trig` | struct | A structure defining the trigger event. Contains fields: type ('trigger'), channel (the selected channel), delay (delay before trigger), and duration (duration of trigger). |

## Examples

```matlab
trig = mr.makeTrigger('physio1', 'delay', 0.01, 'duration', 0.02);
trig = mr.makeTrigger('physio2');
mySys = mr.opts('MaxGrad', 40); trig = mr.makeTrigger('physio1','system',mySys);
```

## See Also

[Sequence.addBlock](addBlock.md), [mr.opts](opts.md)
