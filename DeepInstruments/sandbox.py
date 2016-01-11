import os
os.environ["MEDLEYDB_PATH"] = os.path.expanduser("~/datasets/MedleyDB")

import DeepInstruments as di
import librosa
import matplotlib.pyplot as plt
import numpy as np
import medleydb.sql


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
stems = session.query(medleydb.sql.model.Stem).filter(
    medleydb.sql.model.Track.has_bleed == False
)
(test_stems, training_stems) = di.singlelabel.split_stems(
    di.singlelabel.names, di.singlelabel.test_discarded,
    di.singlelabel.training_discarded, di.singlelabel.training_to_test,
    stems)


# Compute audio features on test set
X_classes = []
for class_stems in training_stems:
    X_files = []
    for stem in class_stems:
        X = di.audio.get_X(decision_length, fmin, hop_length,
                           n_bins_per_octave, n_octaves, stem)
        X_files.append(X)
    X_classes.append(X_files)

activations_classes = []
for class_stems in training_stems:
    activations_files = map(di.singlelabel.get_activation, class_stems)
    activations_classes.append(activations_files)


full_X = np.hstack([np.hstack(X_class) for X_class in X_classes])
X_mean = np.mean(full_X)
X_std = np.std(full_X)

n_classes = len(X_classes)
for class_id in range(n_classes):
    n_files = len(X_classes[class_id])
    for file_id in range(X_classes[class_id]):
        X[class_id][file_id] = (X[class_id][file_id] - X_mean) / X_std


session = medleydb.sql.session()
tracks = session.query(medleydb.sql.model.Track).all()
track = tracks[1]
