# opts

Sets gradient limits and other system properties of the MR system.  It allows users to define or retrieve system parameters such as maximum gradient strength, slew rate, RF pulse parameters, and timing parameters. The function uses an input parser for flexible parameter specification and incorporates default values for system characteristics.

## Syntax

```matlab
function out=opts(varargin)
```

## Calling Pattern

```matlab
mr.opts(...)
```

## Parameters


### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `gradUnit` | string | `validGradUnits{1}` | Specifies the units for maximum gradient amplitude.  Must be one of the validGradUnits. Valid values: ['Hz/m', 'mT/m', 'rad/ms/mm'] | `'mT/m'` |
| `slewUnit` | string | `validSlewUnits{1}` | Specifies the units for maximum slew rate. Must be one of the validSlewUnits. Valid values: ['Hz/m/s', 'mT/m/ms', 'T/m/s', 'rad/ms/mm/ms'] | `'T/m/s'` |
| `b1Unit` | string | `validB1Units{1}` | Specifies the units for maximum B1 amplitude. Must be one of the validB1Units. Valid values: ['Hz', 'T', 'mT', 'uT'] | `'uT'` |
| `maxGrad` | double | `[]` | Specifies the maximum gradient amplitude. (Units: Hz/m) | `40` |
| `maxSlew` | double | `[]` | Specifies the maximum gradient slew rate. (Units: Hz/m/s) | `170` |
| `maxB1` | double | `[]` | Specifies the maximum RF amplitude. (Units: Hz) | `20` |
| `riseTime` | double | `[]` | Specifies the rise time of the gradient waveforms. (Units: seconds) | `0.001` |
| `rfDeadTime` | double | `defaultOpts.rfDeadTime` | Specifies the dead time after an RF pulse. (Units: seconds) | `0` |
| `rfRingdownTime` | double | `defaultOpts.rfRingdownTime` | Specifies the ringdown time after an RF pulse. (Units: seconds) | `0` |
| `adcDeadTime` | double | `defaultOpts.adcDeadTime` | Specifies the dead time after ADC sampling. (Units: seconds) | `0` |
| `adcRasterTime` | double | `defaultOpts.adcRasterTime` | Specifies the raster time for ADC sampling. (Units: seconds) | `100e-9` |
| `rfRasterTime` | double | `defaultOpts.rfRasterTime` | Specifies the raster time for RF pulses. (Units: seconds) | `1e-6` |
| `gradRasterTime` | double | `defaultOpts.gradRasterTime` | Specifies the raster time for gradient waveforms. (Units: seconds) | `10e-6` |
| `blockDurationRaster` | double | `defaultOpts.blockDurationRaster` | Specifies the raster time for a block of events. (Units: seconds) | `10e-6` |
| `adcSamplesLimit` | double | `defaultOpts.adcSamplesLimit` | Specifies the maximum number of ADC samples. | `0` |
| `rfSamplesLimit` | double | `defaultOpts.rfSamplesLimit` | Specifies the maximum number of RF samples. | `0` |
| `adcSamplesDivisor` | double | `defaultOpts.adcSamplesDivisor` | Specifies the divisor for ADC samples. The actual number of samples should be an integer multiple of this divisor. | `4` |
| `gamma` | double | `defaultOpts.gamma` | Specifies the gyromagnetic ratio. (Units: Hz/T) | `42576000` |
| `B0` | double | `defaultOpts.B0` | Specifies the main magnetic field strength. (Units: T) | `1.5` |
| `setAsDefault` | logical | `false` | If true, sets the specified parameters as the new default options. | `true` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `out` | struct | A structure containing the system parameters. |

## Examples

```matlab
mr.opts('maxGrad',30,'gradUnit','mT/m')
mr.opts()
myOpts = mr.opts('maxSlew', 200, 'slewUnit', 'T/m/s');
```

## See Also

[mr.convert](convert.md)
