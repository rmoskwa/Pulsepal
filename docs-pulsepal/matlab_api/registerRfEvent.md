# registerRfEvent

Pre-registers an RF pulse event in the sequence object's library for performance optimization. This method stores RF pulse events in an internal library and returns a unique ID that can be used for fast event retrieval. Additionally, it returns shape IDs for the magnitude, phase, and time components of the RF pulse, which can be reused to avoid redundant shape compression operations.

## Syntax

```matlab
function [id, shapeIDs] = registerRfEvent(event)
```

## Calling Pattern

```matlab
[id, shapeIDs] = seq.registerRfEvent(rfEvent)
```

## Parameters

### Required Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `event` | struct |  | An RF pulse event structure created by makeSincPulse, makeBlockPulse, makeGaussPulse, makeAdiabaticPulse, etc. | `makeSincPulse(pi/2, 'Duration', 1e-3)` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `id` | double | Unique identifier for the registered RF event in the library |
| `shapeIDs` | array | Array of three shape IDs: [magnitude_shape_id, phase_shape_id, time_shape_id] |

## Usage Notes

- Pre-registering RF events is essential for sequences with phase cycling or RF spoiling
- The shapeIDs output can be stored in the RF event structure to avoid recomputing shapes
- The method automatically handles RF pulse compression and normalization
- Required 'use' parameter must be specified in the RF event (excitation, refocusing, inversion, etc.)
- Phase values are automatically normalized to 0-2π range

## Examples

```matlab
% From MPRAGE sequence: pre-register RF events
% Only pre-register shapes for RF that will change phase (RF spoiling)
[~, rf.shapeIDs] = seq.registerRfEvent(rf); % phase will change, only pre-register shapes

% Pre-register complete RF event that won't change
rf180 = mr.makeAdiabaticPulse('hypsec', sys, 'Duration', 10.24e-3, 'use', 'inversion');
[rf180.id, rf180.shapeIDs] = seq.registerRfEvent(rf180);

% RF spoiling implementation from MPRAGE
rfSpoilingInc = 117;  % RF spoiling increment
rf_phase = 0;
rf_inc = 0;

for i = 1:N(ax.n2)  % inner loop
    rf.phaseOffset = rf_phase/180*pi;  % unit: radian
    adc.phaseOffset = rf_phase/180*pi;  % unit: radian
    rf_inc = mod(rf_inc + rfSpoilingInc, 360.0);
    rf_phase = mod(rf_phase + rf_inc, 360.0);

    seq.addBlock(rf);  % Uses pre-registered shapes
end

% From 3D Gradient Echo: register RF for phase cycling
rf = mr.makeBlockPulse(flip*pi/180, 'Duration', Trf, 'system', sys);
[~, rf.shapeIDs] = seq.registerRfEvent(rf); % register shapes only
```

## See Also

[registerGradEvent](registerGradEvent.md), [registerLabelEvent](registerLabelEvent.md), [makeBlockPulse](makeBlockPulse.md), [makeSincPulse](makeSincPulse.md), [makeGaussPulse](makeGaussPulse.md), [makeAdiabaticPulse](makeAdiabaticPulse.md), [addBlock](addBlock.md)
