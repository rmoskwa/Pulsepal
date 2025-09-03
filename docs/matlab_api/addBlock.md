# addBlock

Adds a new block of events to a Pulseq sequence.  It offers three ways to add a block: 1) using a pre-defined block structure; 2) specifying individual events; 3) specifying a duration and then populating the block with events, up to that duration.

## Syntax

```matlab
function addBlock(varargin)
```

## Calling Pattern

```matlab
seq.addBlock(...)
```

## Parameters


### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `blockStruct` | struct |  | A pre-defined block structure | `struct('type', 'adc', 'duration', 0.001)` |
| `duration` | double |  | Optional duration for the block. If provided as first argument, all subsequent events will be added to a block with this duration (Units: seconds) | `0.01` |
| `events` | varargs |  | One or more events (RF, gradient, ADC) to add to the block | `makeTrapezoid('x', 100, 0.001), makeAdc(0.005)` |
| `trajectory_delay` | double | `0` | Delay before starting the trajectory. Only applicable if the input block events include a gradient event (Units: seconds) | `0.001` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `none` | none | This function modifies the Pulseq sequence object in place. It does not return any value. |

## Examples

```matlab
seq.addBlock(struct('type', 'adc', 'duration', 0.001));
seq.addBlock(makeTrapezoid('x', 100, 0.001));
seq.addBlock(0.01, makeTrapezoid('x', 100, 0.001), makeAdc(0.005))
seq.addBlock(makeTrapezoid('x', 100, 0.001), trajectory_delay = 0.002)
```

## See Also

[setBlock](setBlock.md), [makeAdc](makeAdc.md), [makeTrapezoid](makeTrapezoid.md), [makeSincPulse](makeSincPulse.md)
