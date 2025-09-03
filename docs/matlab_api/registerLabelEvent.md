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
% From MPRAGE with GRAPPA: pre-register label events
lblIncPar = mr.makeLabel('INC', 'PAR', 1);
lblResetPar = mr.makeLabel('SET', 'PAR', 0);

% Set PAT scan flags for GRAPPA
lblSetRefScan = mr.makeLabel('SET', 'REF', true);
lblSetRefAndImaScan = mr.makeLabel('SET', 'IMA', true);
lblResetRefScan = mr.makeLabel('SET', 'REF', false);
lblResetRefAndImaScan = mr.makeLabel('SET', 'IMA', false);

% Pre-register all label events
lblSetRefScan.id = seq.registerLabelEvent(lblSetRefScan);
lblSetRefAndImaScan.id = seq.registerLabelEvent(lblSetRefAndImaScan);
lblResetRefScan.id = seq.registerLabelEvent(lblResetRefScan);
lblResetRefAndImaScan.id = seq.registerLabelEvent(lblResetRefAndImaScan);

% Use in GRAPPA acquisition loop
for count = 1:nPEsamp  % outer loop for phase encoding
    % Set PAT labels for every PE line
    if ismember(PEsamp(count), PEsamp_ACS)
        if ismember(PEsamp(count), PEsamp_u)
            seq.addBlock(lblSetRefAndImaScan, lblSetRefScan);
        else
            seq.addBlock(lblResetRefAndImaScan, lblSetRefScan);
        end
    else
        seq.addBlock(lblResetRefAndImaScan, lblResetRefScan);
    end

    % Inner loop with partition encoding counter
    for i = 1:N(ax.n2)
        seq.addBlock(rf, groSp, gpe1, gpe2, lblIncPar);  % Increment PAR after ADC
        seq.addBlock(adc, gro1);
    end
    seq.addBlock(lblResetPar);  % Reset partition counter
end

% Set initial line label
seq.addBlock(mr.makeLabel('SET', 'LIN', PEsamp(1)-1));
```

## See Also

[registerRfEvent](registerRfEvent.md), [registerGradEvent](registerGradEvent.md), [makeLabel](makeLabel.md), [evalLabels](evalLabels.md), [getSupportedLabels](getSupportedLabels.md), [addBlock](addBlock.md)
