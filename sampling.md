**Sampling X,Y, and Z in the MedleyDB dataset**
*5 janvier 2015*

X
* Original is at 44,1kHz, in int16 format
* Available in track.audio_data
* Convert it to float32, normalize by 32768
* librosa hop size is 1024, that is 23 ms.
* decision duration in test set is 2,97s, that is 131072 audio samples, half-overlapping windows.
* training set uses a generator

Y
* Original is at 172Hz, that is 5.8 ms. 256 audio samples.
* Available at track.activations_data. Columns are stems
* We subsample it by a factor 16. We get a hop size of 4096 samples, that is 93 ms.

Z
* Original is at 172Hz, that is 5.8 ms. 256 audio samples.
* Stems' names can be retrieved by stem.instrument
* We subsample it by a factor 16. We get a hop size of 4096 samples, that is 93 ms.
* Original yields f0 in Hz. We convert it (with librosa tools) to MIDI index, and quantize to ConvNet index. For this, we need: fmin, n_filters_per_octave, pool1_height.


