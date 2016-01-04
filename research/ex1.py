import DeepInstruments as di
import joblib
import sacred

ex = sacred.Experiment("solosdb")


@ex.config
def config():
    batch_size = 32
    decision_duration = 2.048 # in seconds
    epoch_size = 4096
    every_n_epoch = 1
    fmin = 55 # in Hz
    hop_duration = 0.016
    instrument_list = ['Cl', 'Co']
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
    pool2_height = 4
    pool2_height = 8
    dense1_channels = 512
    drop1_proportion = 0.25
    dense2_channels = 64
    drop2_proportion = 0.25


@ex.automain
def delayed_run(batch_size, decision_duration, epoch_size, every_n_epoch,
                fmin, hop_duration, instrument_list, n_bins_per_octave,
                n_epochs, n_octaves, optimizer, silence_threshold,
                sr, test_dirs, train_dirs,
                conv1_channels, conv1_height, conv1_width,
                pool1_width, pool1_height,
                conv2_channels, conv2_height, conv2_width,
                pool2_width, pool2_height,
                dense1_channels, drop1_proportion,
                dense2_channels, drop2_proportion):
    return joblib.delayed(di.main.run)(
        batch_size, decision_duration, epoch_size, every_n_epoch,
        fmin, hop_duration, instrument_list, n_bins_per_octave,
        n_epochs, n_octaves, optimizer, silence_threshold,
        sr, test_dirs, train_dirs,
        conv1_channels, conv1_height, pool1_width, pool1_height,
        conv2_channels, conv2_height, pool2_width, pool2_height,
        dense1_channels, drop1_proportion,
        dense2_channels, drop2_proportion)