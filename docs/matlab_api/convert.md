# convert

Converts numerical data between different units used in magnetic resonance imaging (MRI) gradient and RF pulse design.  It handles units for magnetic field strength (B1), gradient strength, and gradient slew rate.  The function utilizes a pre-defined set of valid units and converts the input data to a standard unit before converting it to the desired output unit. The gyromagnetic ratio (gamma) is used for conversions involving magnetic field strength.

## Syntax

```matlab
function out=convert(in,varargin)
```

## Calling Pattern

```matlab
mr.convert(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `in` | double | The numerical data to be converted. The units of this data are specified by the 'fromUnit' parameter. | `1000` | varies |
| `fromUnit` | char | A string specifying the units of the input data 'in'. | `'mT/m'` |  |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `toUnit` | char | `[]` | A string specifying the desired units for the output data. If omitted, a default unit is chosen based on 'fromUnit'. Valid values: 'Hz', 'T', 'mT', 'uT', 'Hz/m', 'mT/m', 'rad/ms/mm', 'Hz/m/s', 'mT/m/ms', 'T/m/s', 'rad/ms/mm/ms' | `'Hz/m'` |
| `gamma` | double | `42.576e6` | The gyromagnetic ratio, used for conversions involving magnetic field strength. Defaults to 42.576 MHz/T (for protons). (Units: Hz/T) | `42.577e6` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `out` | double | The converted numerical data in the specified 'toUnit' units. |

## Examples

```matlab
out = mr.convert(1000,'mT/m','Hz/m');
out = mr.convert(500,'rad/ms/mm','mT/m');
out = mr.convert(200,'Hz',[], 42.58e6);
```
