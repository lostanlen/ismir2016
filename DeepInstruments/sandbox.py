import os
os.environ["MEDLEYDB_PATH"] = os.path.expanduser("~/datasets/MedleyDB")

import DeepInstruments as di
import librosa
import matplotlib.pyplot as plt
import numpy as np
import medleydb.sql

session = medleydb.sql.session()
stems = session.query(medleydb.sql.model.Stem).all()
stems = [ stem for stem in stems if not stem.track.has_bleed]


batch_size = 32
decision_length = 131072 # in samples
epoch_size = 4096
every_n_epoch = 1
fmin = 55 # in Hz
hop_length = 1024 # in samples
n_bins_per_octave = 12
n_epochs = 1
n_octaves = 8
optimizer = "adagrad"
silence_threshold = -0.7
sr = 32000
test_dirs = ["~/datasets/solosDb8/test"]
train_dirs = ["~/datasets/solosDb8/train"]
conv1_channels = 100
conv1_height = 96
conv1_width = 32
pool1_height = 7
pool1_width = 7
conv2_channels = 100
conv2_height = 8
conv2_width = 8
pool2_height = 8
dense1_channels = 512
drop1_proportion = 0.5
dense2_channels = 64
drop2_proportion = 0.5


session = medleydb.sql.session()
tracks = session.query(medleydb.sql.model.Track).all()
track = tracks[1]

# X (audio representation)
X = di.audio.get_X(decision_length, fmin, hop_length, n_bins_per_octave,
                   n_octaves, track)

# Y (all instrument activations)
activations = di.wrangling.get_activations(instrument_names, track)


# Melodic Z (piano-rolls, i.e. time-frequency activations)
pianorolls = di.wrangling.get_pianorolls(fmin, melodic_names,
                                         n_bins_per_octave, n_octaves, track)


# Non-melodic Z (non-melodic activations)
nonmelodic_activations = di.wrangling.get_activations(nonmelodic_names, track)