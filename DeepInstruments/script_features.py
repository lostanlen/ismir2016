# Compute audio features and retrieve melodies on the training set
datagen = di.singlelabel.ScalogramGenerator(
        decision_length, fmin, hop_length, Q, n_octaves, training_stems)

# Compute audio features on the test set
test_paths = di.singlelabel.get_paths("test")
X_test = datagen.get_X(test_paths)
y_test = np.hstack(map(di.descriptors.get_y, test_paths))
