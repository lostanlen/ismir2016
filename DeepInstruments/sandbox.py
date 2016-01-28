import DeepInstruments as di
import librosa
import matplotlib.pyplot as plt
import numpy as np
import sklearn.ensemble

batch_size = 512
decision_length = 131072  # in samples
epoch_size = 8192
every_n_epoch = 10
fmin = 55 # in Hz
hop_length = 1024 #  in samples
n_bins_per_octave = 12
n_epochs = 20
n_octaves = 8
sr = 32000
conv1_channels = 100
conv1_height = 96
conv1_width = 32
pool1_height = 3
pool1_width = 3
conv2_channels = 100
conv2_height = 8
conv2_width = 8
pool2_height = 8
dense1_channels = 512
drop1_proportion = 0.5
dense2_channels = 64
drop2_proportion = 0.5


(test_stems, training_stems) = di.singlelabel.get_stems()

datagen = di.singlelabel.ScalogramGenerator(decision_length, fmin,
                                            hop_length, n_bins_per_octave,
                                            n_octaves, training_stems)