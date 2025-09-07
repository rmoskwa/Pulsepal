# registerLabelEvent

Pre-registers a label event in the sequence object's library for performance optimization. This method stores label events (SET or INC operations) in an internal library and returns a unique ID for fast retrieval. Labels are used for tracking counters, flags, and states during sequence execution, particularly important for parallel imaging (GRAPPA/SENSE) and multi-contrast acquisitions.

## Syntax

```matlab
function id = registerLabelEvent(event)
```

## Calling Pattern

```matlab
id = seq.registerLabelEvent(labelEvent)
```

## Parameters

### Required Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `event` | struct |  | A label event structure created by makeLabel | `makeLabel('INC', 'LIN', 1)` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `id` | double | Unique identifier for the registered label event in the library |

## Usage Notes

- Labels are essential for parallel imaging sequences (GRAPPA, SENSE) to mark reference lines
- Common labels include: LIN (line number), PAR (partition), REF (reference scan), IMA (image scan)
- Two operation types: 'SET' (sets a value) and 'INC' (increments by a value)
- Pre-registering label events improves performance in sequences with complex sampling patterns
- Labels must be from the list of supported labels (use getSupportedLabels() to see available options)

## Examples

```matlab
% GRAPPA reference and image scan labels
lblSetRefScan = mr.makeLabel('SET','REF', true);
lblSetRefAndImaScan = mr.makeLabel('SET','IMA', true);
lblResetRefScan = mr.makeLabel('SET','REF', false);
lblResetRefAndImaScan = mr.makeLabel('SET','IMA', false);

lblSetRefScan.id = seq.registerLabelEvent(lblSetRefScan);
lblSetRefAndImaScan.id = seq.registerLabelEvent(lblSetRefAndImaScan);
lblResetRefScan.id = seq.registerLabelEvent(lblResetRefScan);
lblResetRefAndImaScan.id = seq.registerLabelEvent(lblResetRefAndImaScan);

% Performance optimization: pre-register all label events
lblIncLin = mr.makeLabel('INC','LIN', 1);
lblIncPar = mr.makeLabel('INC','PAR', 1);
lblResetPar = mr.makeLabel('SET','PAR', 0);

lblIncLin.id = seq.registerLabelEvent(lblIncLin);
lblIncPar.id = seq.registerLabelEvent(lblIncPar);
lblResetPar.id = seq.registerLabelEvent(lblResetPar);

% Multi-slice EPI labels
lblResetSLC = mr.makeLabel('SET', 'SLC', 0);
lblSetNAV = mr.makeLabel('SET','NAV', 1);
lblIncREP = mr.makeLabel('INC', 'REP', 1);

lblResetSLC.id = seq.registerLabelEvent(lblResetSLC);
lblSetNAV.id = seq.registerLabelEvent(lblSetNAV);
lblIncREP.id = seq.registerLabelEvent(lblIncREP);
```

## See Also

[registerRfEvent](registerRfEvent.md), [registerGradEvent](registerGradEvent.md), [makeLabel](makeLabel.md), [evalLabels](evalLabels.md), [getSupportedLabels](getSupportedLabels.md), [addBlock](addBlock.md)
