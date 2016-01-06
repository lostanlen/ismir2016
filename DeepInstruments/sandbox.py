import DeepInstruments as di
import librosa
import numpy as np
import os
os.environ["MEDLEYDB_PATH"] = os.path.expanduser("~/datasets/MedleyDB")
import medleydb.sql

melodic_names = [u'clean electric guitar',
                 u'female singer',
                 u'violin']

nonmelodic_names = [u'acoustic guitar']

instrument_names = di.wrangling.union(melodic_names, nonmelodic_names)

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
track = tracks[0]


X = di.audio.get_X(decision_length,
                   fmin,
                   hop_length,
                   n_bins_per_octave,
                   n_octaves,
                   track)

activations = di.wrangling.get_activations(instrument_names, track)

stems = track.stems.all()
ranks = [ stem.rank for stem in stems ]



melody0 = track.melodies[0]
melody0 = np.vstack(melody0.annotation_data)[:,1:]

n_bins = n_bins_per_octave * n_octaves
freqs = librosa.cqt_frequencies(bins_per_octave=n_bins_per_octave,
                                fmin=fmin,
                                n_bins=n_bins)
midis = librosa.hz_to_midi(freqs)
# for melody in track.melodies
melody = track.melodies[0]

melody_f0s = np.vstack(melody.annotation_data)[:, 1:]
melody_pitches = librosa.hz_to_midi(melody_f0s)
melody_pitches[np.isinf(melody_pitches)] = 0.0




matching = [ s in instrument_list for s in stem_instruments]
n_frames = len(y)
