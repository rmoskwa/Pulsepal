# makeBlockPulse

Generates a block RF pulse with optional slice-selective capabilities.  The function creates a Pulseq RF pulse object defining the characteristics of the pulse, including amplitude, duration, frequency and phase offsets. It can calculate the duration based on either the provided bandwidth or time-bandwidth product.  It also handles optional parameters for slice selection (gradients) and returns both the RF pulse and a delay object to account for ringdown time.

## Syntax

```matlab
function [rf, delay] = makeBlockPulse(flip,varargin)
```

## Calling Pattern

```matlab
mr.makeBlockPulse(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `flip` | double | The flip angle of the RF pulse. | `pi/2` | radians |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `system` | struct | `[]` | System parameters structure (e.g., mr.opts()).  Provides parameters like rfRasterTime, rfDeadTime, rfRingdownTime needed for pulse generation. If empty, default system parameters are used. | `mr.opts()` |
| `duration` | double | `0` | Duration of the RF pulse. If 0, it will be calculated from bandwidth or timeBwProduct. (Units: seconds) | `0.004` |
| `freqOffset` | double | `0` | Frequency offset of the RF pulse (Hz). (Units: Hz) | `100` |
| `phaseOffset` | double | `0` | Phase offset of the RF pulse (radians). (Units: radians) | `pi/4` |
| `freqPPM` | double | `0` | Frequency offset in parts per million (ppm) relative to the Larmor frequency. Can be used in addition to freqOffset. (Units: ppm) | `10` |
| `phasePPM` | double | `0` | Phase offset in parts per million (ppm) relative to the Larmor frequency. Can be used in addition to phaseOffset. (Units: ppm) | `5` |
| `timeBwProduct` | double | `0` | Time-bandwidth product of the RF pulse.  If greater than zero, duration is calculated as timeBwProduct/bandwidth.  | `1` |
| `bandwidth` | double | `0` | Bandwidth of the RF pulse (Hz). If greater than zero, duration is calculated as 1/(4*bandwidth). (Units: Hz) | `2500` |
| `maxGrad` | double | `0` | Maximum gradient amplitude for slice selection (Hz/m). (Units: Hz/m) | `50e6` |
| `maxSlew` | double | `0` | Maximum gradient slew rate for slice selection (Hz/m/s). (Units: Hz/m/s) | `200e6` |
| `sliceThickness` | double | `0` | Slice thickness for slice selection (meters). (Units: meters) | `0.005` |
| `delay` | double | `0` | Additional delay after the RF pulse (seconds). (Units: seconds) | `0.001` |
| `use` | char | `'u'` | Specifies the pulse type; affects k-space trajectory calculation.  Must be one of mr.getSupportedRfUse(). Valid values: mr.getSupportedRfUse() | `'excitation'` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `rf` | struct | Pulseq RF pulse object containing the generated RF pulse parameters. |
| `delay` | struct | Pulseq delay object accounting for RF ringdown time. |

## Examples

```matlab
rf = mr.makeBlockPulse(pi/2, 'duration', 0.001);
[rf, delay] = mr.makeBlockPulse(pi/4, 'bandwidth', 1000, 'freqOffset', 100, 'system', mr.opts());
```

## See Also

[Sequence.addBlock](addBlock.md), [mr.opts](opts.md), [mr.getSupportedRfUse](getSupportedRfUse.md)
