# makeDigitalOutputPulse

Creates a digital output pulse event, also known as a trigger, for a specified channel.  This function generates a structure defining the trigger parameters to be used within a Pulseq sequence.

## Syntax

```matlab
function trig = makeDigitalOutputPulse(channel, varargin)
```

## Calling Pattern

```matlab
mr.makeDigitalOutputPulse(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `channel` | char | Specifies the output channel for the trigger.  Valid values are 'osc0', 'osc1', and 'ext1'. | `'osc0'` |  |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `delay` | double | `0` | Specifies the delay before the trigger starts, in seconds. (Units: seconds) | `0.001` |
| `duration` | double | `0` | Specifies the duration of the trigger pulse, in seconds. If smaller than the system's gradRasterTime, it is set to gradRasterTime. (Units: seconds) | `0.002` |
| `system` | struct | `[]` | A structure containing system parameters.  If empty, defaults to mr.opts().  Should contain at least gradRasterTime. | `mr.opts()` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `trig` | struct | A structure containing the trigger parameters: type ('output'), channel, delay, and duration. |

## Examples

```matlab
trig = mr.makeDigitalOutputPulse('osc0', 'delay', 0.001, 'duration', 0.002);
trig = mr.makeDigitalOutputPulse('ext1', 'delay', 0.01);
mySys = mr.opts('gradRasterTime', 0.0001); trig = mr.makeDigitalOutputPulse('osc1', 'system', mySys);
```

## See Also

[Sequence.addBlock](addBlock.md)
