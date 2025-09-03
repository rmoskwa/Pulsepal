# sound

Plays the sequence waveforms through the system speaker.  It processes the waveforms from the Pulseq sequence object, applies channel weighting, performs interpolation to match the desired sample rate, applies a Gaussian filter to suppress ringing artifacts, normalizes the amplitude, and then plays the resulting audio.  The function allows specifying a range of blocks to play and provides an option to only generate the sound data without playing it.

## Syntax

```matlab
function soundData=sound(varargin)
```

## Calling Pattern

```matlab
seq.sound(...)
```

## Parameters


### Optional Parameters

| Name | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `blockRange` | double | `[1 inf]` | Specifies the range of blocks in the sequence to play.  Defaults to playing all blocks. Valid values: A two-element numeric vector [start_block end_block].  end_block can be Inf to indicate the end of the sequence. | `[5, 10]` |
| `channelWeights` | double | `[1 1 1]` | Specifies the weights for the three channels (x, y, z). These weights scale the amplitudes of the corresponding channel waveforms before combining them for playback. Valid values: A three-element numeric vector [weight_x weight_y weight_z]. | `[0.8, 1.2, 0.5]` |
| `onlyProduceSoundData` | logical | `false` | If true, the function only produces the sound data without actually playing it. This is useful if you want to process or save the sound data before playback. Valid values: true or false | `true` |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `soundData` | double | A 2xN matrix containing the processed sound data for two channels.  Each column represents a sample.  If onlyProduceSoundData is true, this is the only output. |

## Examples

```matlab
soundData = seq.sound(); % Plays the entire sequence
soundData = seq.sound('blockRange', [10, 20]); % Plays blocks 10-20
seq.sound('channelWeights', [0.5, 1, 0], 'onlyProduceSoundData', true); % Generates sound data without playing, weighting channels differently
```

## See Also

[waveforms_and_times](waveforms_and_times.md)
