# simRf

Simulates the effect of an RF pulse on nuclear magnetization using a quaternion-based rotation formalism.  It takes a Pulseq RF pulse definition as input and returns the resulting magnetization components, frequency axis, and refocusing efficiency.

## Syntax

```matlab
function [Mz_z,Mz_xy,F,ref_eff,Mx_xy,My_xy]=simRf(rf,rephase_factor,prephase_factor)
```

## Calling Pattern

```matlab
mr.simRf(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `rf` | struct | A Pulseq RF pulse structure.  This structure contains all the parameters defining the RF pulse, such as amplitude, duration, shape, frequency, phase, etc. | `Pulseq RF pulse structure (created using Pulseq functions)` |  |

### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `rephase_factor` | double | `0` | A factor used to correct the phase of the magnetization, particularly important for slice-selective excitation pulses.  It accounts for the timing difference between the pulse's center and its duration. Valid values: Any real number | `0.004` |
| `prephase_factor` | double | `0` | An experimental parameter used for simulating refocusing pulses or spoiling.  It allows for additional phase adjustments beyond the rephase factor. Valid values: Any real number | `0` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `Mz_z` | double | The z-component of the magnetization after the RF pulse, assuming initial magnetization along the z-axis. |
| `Mz_xy` | double | The xy-component of the magnetization after the RF pulse, assuming initial magnetization along the z-axis. |
| `F` | double | The frequency axis in Hz. |
| `ref_eff` | double | The refocusing efficiency of the pulse as a complex number. The magnitude reflects the refocusing and the phase is related to the effective phase of the RF pulse. |
| `Mx_xy` | double | The xy-component of the magnetization after the RF pulse, assuming initial magnetization along the x-axis. |
| `My_xy` | double | The xy-component of the magnetization after the RF pulse, assuming initial magnetization along the y-axis. |

## Examples

```matlab
% RF profile validation for excitation pulse
[M_z90, M_xy90, F2_90] = mr.simRf(rf_ex);
sl_th_90 = mr.aux.findFlank(F2_90(end:-1:1)/gz.amplitude, abs(M_xy90(end:-1:1)), 0.5) - mr.aux.findFlank(F2_90/gz.amplitude, abs(M_xy90), 0.5);

% RF profile validation for refocusing pulse
[M_z180, M_xy180, F2_180, ref_eff] = mr.simRf(rf_ref);
sl_th_180 = mr.aux.findFlank(F2_180(end:-1:1)/g_ref.amplitude, abs(ref_eff(end:-1:1)), 0.5) - mr.aux.findFlank(F2_180/g_ref.amplitude, abs(ref_eff), 0.5);

% Basic RF simulation
[Mz_z, Mz_xy, F, ref_eff, Mx_xy, My_xy] = mr.simRf(rf_pulse_struct);
```

## See Also

[mr.calcRfBandwidth](calcRfBandwidth.md)
