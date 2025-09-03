# registerGradEvent

Pre-registers a gradient event in the sequence object's library for performance optimization. This method stores gradient events (trapezoids or arbitrary gradients) in an internal library and returns a unique ID that can be used for fast event retrieval in loops. This significantly accelerates sequence assembly when the same gradient event is used multiple times.

## Syntax

```matlab
function [id, shapeIDs] = registerGradEvent(event)
```

## Calling Pattern

```matlab
[id, shapeIDs] = seq.registerGradEvent(gradientEvent)
```

## Parameters

### Required Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `event` | struct |  | A gradient event structure created by makeTrapezoid, makeArbitraryGrad, or scaleGrad | `makeTrapezoid('x', 'Area', 1e-4)` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `id` | double | Unique identifier for the registered gradient event in the library |
| `shapeIDs` | array | Array of shape IDs for gradient waveform components (optional output) |

## Usage Notes

- Pre-registering gradient events is particularly useful in loops where the same gradient is used repeatedly
- The returned ID can be stored in the event structure as `event.id` for faster addBlock operations
- This method automatically handles gradient shape compression for efficient storage
- Supports both trapezoidal and arbitrary gradient waveforms

## Examples

```matlab
% From MPRAGE sequence: pre-register objects that do not change while looping
gslSp.id = seq.registerGradEvent(gslSp);
groSp.id = seq.registerGradEvent(groSp);
gro1.id = seq.registerGradEvent(gro1);

% Pre-register phase-encoding gradients that repeat in the inner loop
gpe2je = mr.scaleGrad(gpe2, pe2Steps(PEsamp(count)));
gpe2je.id = seq.registerGradEvent(gpe2je);
gpe2jr = mr.scaleGrad(gpe2, -pe2Steps(PEsamp(count)));
gpe2jr.id = seq.registerGradEvent(gpe2jr);

% Use pre-registered gradients in sequence assembly
for i = 1:N(ax.n2)  % inner loop for partition encoding
    if (i == 1)
        seq.addBlock(rf);
    else
        seq.addBlock(rf, groSp, mr.scaleGrad(gpe1, -pe1Steps(i-1)), gpe2jr);
    end
    seq.addBlock(adc, gro1, mr.scaleGrad(gpe1, pe1Steps(i)), gpe2je);
end
```

## See Also

[registerRfEvent](registerRfEvent.md), [registerLabelEvent](registerLabelEvent.md), [makeTrapezoid](makeTrapezoid.md), [makeArbitraryGrad](makeArbitraryGrad.md), [scaleGrad](scaleGrad.md), [addBlock](addBlock.md)
