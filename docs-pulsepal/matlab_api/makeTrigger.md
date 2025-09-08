# makeTrigger

Creates a trigger event for synchronization with an external signal.  This function generates a structure defining a trigger event for use within a Pulseq sequence.  It specifies the trigger channel, delay before the trigger, and duration after the trigger.  The duration is constrained to be at least as long as the system's gradient raster time.

## Syntax

```matlab
function trig = makeTrigger(channel, varargin)
```

## Calling Pattern

```matlab
mr.makeTrigger(...)
mr.makeTrigger('ParameterName', value, ...)
```

## Parameters

This function accepts both positional parameters and name-value pairs.

### Required Parameters

| Parameter Name | Value Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `channel` | char | The channel to use for the trigger.  Valid values are 'physio1' and 'physio2' (Siemens-specific). | `'physio1'` |  |

### Name-Value Pair Arguments
| Parameter Name (string) | Value Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `delay` | double | `0` | The delay (in seconds) before the trigger event starts. (Units: seconds) | `0.001` |
| `duration` | double | `0` | The duration (in seconds) of the trigger event.  If shorter than the system's gradient raster time, it is automatically adjusted to match the raster time. (Units: seconds) | `0.005` |
| `system` | struct | `[]` | A structure containing system parameters (e.g., from mr.opts()). If not provided, mr.opts() is used. | `mr.opts()` |

## Returns

| Output | Value Type | Description |
|--------|------|-------------|
| `trig` | struct | A structure defining the trigger event. Contains fields: type ('trigger'), channel (the selected channel), delay (delay before trigger), and duration (duration of trigger). |

## Examples

```matlab
% Physiological trigger for cardiac gating with extended duration
trig = mr.makeTrigger('physio1','duration', 2000e-6);

% Simple physiological trigger on channel 1
trig = mr.makeTrigger('physio1');

% Trigger with custom system parameters
sys = mr.opts('gradRasterTime', 10e-6);
trig = mr.makeTrigger('physio2','system', sys);
```

## See Also

[Sequence.addBlock](addBlock.md), [mr.opts](opts.md)
