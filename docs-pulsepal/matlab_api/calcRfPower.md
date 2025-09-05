# calcRfPower

Calculates the relative power of an RF pulse.  It returns the relative energy of the pulse in units of Hz^2*s (which simplifies to Hz), where the units represent RF amplitude squared multiplied by time.  The `mean_pwr` output is closely related to relative SAR.  The function also returns the peak power (in Hz^2) and the RMS B1 amplitude (in Hz).  The power and amplitude values are relative; to convert the RF amplitude to Tesla (T), divide by the gyromagnetic ratio (γ).  Similarly, to convert the power to mT^2*s, divide by γ^2. Note that absolute SAR calculation requires additional coil and subject-dependent scaling factors.

## Syntax

```matlab
function [mean_pwr, peak_pwr, rf_rms, total_energy]=calcRfPower(varargin)
```

## Calling Pattern

```matlab
seq.calcRfPower(...)
```

## Parameters


### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `blockRange` | double | `[1 inf]` | Specifies the range of blocks in the sequence for which to calculate the power.  Defaults to the entire sequence. Valid values: A two-element numeric vector [start_block, end_block] | `[10, 20]` |
| `windowDuration` | double | `NaN` | Specifies the time window for calculating total_energy, mean_pwr, and rf_rms. If provided, the function returns the maximum values over all time windows. The window duration is rounded up to a certain number of complete blocks. Valid values: A positive numeric scalar (Units: seconds) | `0.005` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `mean_pwr` | double | Mean power of the RF pulse (relative, in Hz). |
| `peak_pwr` | double | Peak power of the RF pulse (relative, in Hz^2). |
| `rf_rms` | double | RMS B1 amplitude of the RF pulse (relative, in Hz). |
| `total_energy` | double | Total energy of the RF pulse (relative, in Hz). |

## Examples

```matlab
seq.calcRfPower()
seq.calcRfPower('blockRange', [10, 20])
seq.calcRfPower('windowDuration', 0.01)
```

## See Also

[mr.calcRfPower](calcRfPower.md)
