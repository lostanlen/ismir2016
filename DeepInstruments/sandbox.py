import DeepInstruments as di
import numpy as np
import os
os.environ["MEDLEYDB_PATH"] = os.path.expanduser("~/datasets/MedleyDB")
import medleydb.sql

batch_size = 32
decision_length = 131072 # in samples
epoch_size = 4096
every_n_epoch = 1
fmin = 55 # in Hz
hop_length = 1024 # in samples
instrument_names = [u'female singer', u'acoustic guitar', u'violin']
n_bins_per_octave = 16
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
track = session.query(medleydb.sql.model.Track).first()

X = di.audio.get_X(decision_length,
                   fmin,
                   hop_length,
                   n_bins_per_octave,
                   n_octaves,
                   track)

stem_activations = track.activations_data
stem_activations = np.vstack(stem_activations)
stem_names = [s.instrument.name for s in track.stems]

instrument_matches = di.wrangling.instrument_stems(instrument_names,
                                                   stem_names)
n_instruments = len(instrument_names)

for instrument_index in range(n_instruments):
    instrument_match = instrument_matches[instrument_index]
    instrument_activations = stem_activations[instrument_match]




matching = [ s in instrument_list for s in stem_instruments]
n_frames = len(y)
