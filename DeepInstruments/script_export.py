import matplotlib.pyplot as plt
import numpy as np
import librosa

# Save results
np.savez(
    export_str + ".npz",
    decision_length=decision_length,
    fmin=fmin,
    hop_length=hop_length,
    n_bins_per_octave=n_bins_per_octave,
    n_octaves=n_octaves,
    conv1_channels=conv1_channels,
    conv1_height=conv1_height,
    conv1_width=conv1_width,
    pool1_height=pool1_height,
    pool1_width=pool1_width,
    conv2_channels=conv2_channels,
    conv2_height=conv2_height,
    conv2_width=conv2_width,
    pool2_height=pool2_height,
    pool2_width=pool2_width,
    dense1_channels=dense1_channels,
    batch_size=batch_size,
    epoch_size=epoch_size,
    n_epochs=n_epochs,
    optimizer=optimizer,
    chunk_accuracies_history=chunk_accuracies_history,
    file_accuracies_history=file_accuracies_history,
    final_chunk_score=final_chunk_score,
    final_mean_chunk_score=final_mean_chunk_score,
    final_file_score=final_file_score,
    final_mean_file_score=final_mean_file_score)

# Save weights
graph.save_weights(export_str + ".h5", overwrite=True)

# Save images for first-layer kernels
if is_spiral:
    for j in range(6):
        octave_index = 6 * j + 4
        octave = graph.get_weights()[octave_index]
        kernels = [octave[i, 0, :, :] for i in range(conv1_channels)]
        zero_padding = 0.0 * np.ones((conv1_height, 1))
        kernels = [np.concatenate((kernel, zero_padding), axis=1)
                   for kernel in kernels]
        kernels = np.hstack(kernels)
        librosa.display.specshow(kernels)
        plt.savefig(export_str + "-j" + str(j) + ".png")
else:
    first_layer = graph.get_weights()[0]
    kernels = [first_layer[i, 0, :, :] for i in range(conv1_channels)]
    zero_padding = -0.0 * np.ones((conv1_height, 1))
    registered_kernels = [np.concatenate((kernel, zero_padding), axis=1)
                          for kernel in kernels]
    librosa.display.specshow(np.hstack(kernels))
    plt.savefig(export_str + ".png")

# For Fig 3
import theano
example_id = 11000
X = X_test[example_id:(example_id+1), :, :, :] - 0.33
librosa.display.specshow(X[0, 0, :, :])
plt.savefig("Fig3_X.png")
pool1_f =\
    theano.function([graph.get_input(train=False)],
                     graph.nodes["pool1_2d"].get_output(train=False))
pool1_activations = pool1_f(X)[0]

for i in range(conv1_channels[0]):
    librosa.display.specshow(pool1_activations[i, :, :])
    plt.savefig("Fig3_pool1_" + str(i) + ".png")



pool2_f =\
    theano.function([graph.get_input(train=False)],
                     graph.nodes["pool2_2d"].get_output(train=False))
pool2_activations = pool2_f(X)[0]
for i in range(conv2_channels[0]):
    librosa.display.specshow(pool2_activations[i, :, :])
    plt.savefig("Fig3_pool2_" + str(i) + ".png")
