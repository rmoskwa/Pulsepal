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
% Standard trigger pulse for EPI sequences
trig = mr.makeDigitalOutputPulse('osc0','duration', 100e-6);

% External trigger with delay for cardiac gating
trig_out = mr.makeDigitalOutputPulse('ext1','duration', 100e-6,'delay',500e-6);

% Multiple trigger channels available: 'osc0','osc1','ext1'
trig = mr.makeDigitalOutputPulse('osc1','duration', 200e-6);
```

## See Also

[Sequence.addBlock](addBlock.md)
