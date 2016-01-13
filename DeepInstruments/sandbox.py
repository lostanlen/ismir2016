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


# Compute audio features X
X_classes = []
for class_stems in training_stems:
    X_files = []
    for stem in class_stems:
        X = di.audio.get_X(decision_length, fmin, hop_length,
                           n_bins_per_octave, n_octaves, stem)
        X_files.append(X)
    X_classes.append(X_files)


# Compute activations Y
Y_classes = []
for class_stems in training_stems:
    Y_files = []
    for stem in class_stems:
        print(stem.track.artist)
        print(stem.track.name)
        Y_files.append(di.singlelabel.get_Y(stem))
    Y_classes.append(Y_files)


stems = training_stems
# Find indices of activated instruments
indices_classes = di.singlelabel.get_indices(stems, decision_length)


# Get melodies
melody_classes = []
for class_stems in training_stems:
    melody_files = map(di.singlelabel.get_melody, class_stems)
    melody_classes.append(melody_files)


# Draw a chunk at random
n_classes = len(training_stems)
half_activation_hop = int(0.5 * (float(decision_length) / 2048))
random_class = np.random.randint(n_classes)
indices_files = indices_classes[random_class]
file_lengths = map(len, indices_files)
file_probabilities = map(float, file_lengths) / np.sum(file_lengths)
n_files = len(indices_files)
random_file = np.random.choice(n_files, p=file_probabilities)
indices = indices_files[random_file]
random_index = np.random.choice(indices)


# Get X
X_middle = int(random_index * float(hop_length) / 2048)
half_X_hop = int(0.5 * float(decision_length) / hop_length)
X_start = X_middle - half_X_hop
X_stop = X_middle + half_X_hop
X_range = xrange(X_middle-half_X_hop, X_middle+half_X_hop)
X = X_classes[random_class][random_file][:, X_range]

# Get Y
Y = Y_classes[random_class][random_file][:, random_index]

# Get corresponding activation
activation_start = random_index - half_activation_hop
activation_stop = random_index + half_activation_hop
activation_range = range(activation_start, activation_stop)
activation_class = activations_classes[random_class]
activation_file = activation_class[random_file]
activation = activation_file[activation_range]


# Get corresponding melody
melody_start = activation_start * 2048 / 256
melody_stop = activation_stop * 2048 / 256
melody_range = range(melody_start, melody_stop)
melody_class = melody_classes[random_class]
melody_file = melody_class[random_file]
melody = melody_file[melody_range]
melody_gate = np.greater(melody, 0)

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
