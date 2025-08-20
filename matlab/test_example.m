% Test example for PulsePal code awareness
% This is a simple spin echo sequence with potential issues

% Sequence parameters
seq = mr.Sequence();
fov = 256e-3;      % Field of view in meters
Nx = 256;          % Number of readout points
Ny = 256;          % Number of phase encoding steps
TE = 50e-3;        % Echo time (might be too long?)
TR = 500e-3;       % Repetition time

% RF pulse - 90 degree excitation
flip_angle = 90 * pi/180;  % Convert to radians
rf_duration = 2e-3;
rf = mr.makeBlockPulse(flip_angle, 'Duration', rf_duration);

% Slice selection gradient
slice_thickness = 5e-3;  % 5mm slice
gz_amplitude = rf.bandwidth / (gamma * slice_thickness);  % Is this correct?

% The gradient seems too strong - causing issues
gz = mr.makeTrapezoid('z', 'FlatArea', gz_amplitude * rf_duration, ...
                       'FlatTime', rf_duration);

% Readout gradient (potential issue with timing?)
delta_k = 1 / fov;
kWidth = Nx * delta_k;
readout_time = 6.4e-3;

% This gradient calculation might be wrong
gx = mr.makeTrapezoid('x', 'FlatArea', kWidth, 'FlatTime', readout_time);

% ADC
adc = mr.makeAdc(Nx, 'Duration', readout_time, 'Delay', gx.riseTime);

% Add blocks to sequence
seq.addBlock(rf, gz);  % RF with slice select
seq.addBlock(mr.makeDelay(TE/2));  % Is this the right delay?
seq.addBlock(gx, adc);  % Readout

% Problem: The sequence produces artifacts and the flip angle seems wrong
% Question: Why is my flip angle not achieving 90 degrees and what's causing the artifacts?