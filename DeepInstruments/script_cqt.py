import DeepInstruments as di
import numpy as np

# Parameters for audio
decision_length = 131072  # in samples
fmin = 55  # in Hz
hop_length = 1024  # in samples
Q = 12
n_octaves = 8

# Get single-label split (MedleyDB for training, solosDb for test
(test_stems, training_stems) = di.singlelabel.get_stems()

# Compute audio features and retrieve melodies on the training set
datagen = di.singlelabel.ScalogramGenerator(
        decision_length, fmin, hop_length, Q, n_octaves, training_stems)

# Compute audio features on the test set
test_paths = di.singlelabel.get_paths("test")
X_test = datagen.get_X(test_paths)
y_test = np.hstack(map(di.descriptors.get_y, test_paths))
