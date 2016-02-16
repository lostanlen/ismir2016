import DeepInstruments as di


def build_graph(
        is_spiral,
        is_Z_supervision,
        n_bins_per_octave,
        n_octaves,
        X_width,
        conv1_channels,
        conv1_height,
        conv1_width,
        pool1_height,
        pool1_width,
        conv2_channels,
        conv2_height,
        conv2_width,
        pool2_height,
        pool2_width,
        drop1_proportion,
        dense1_channels,
        drop2_proportion,
        dense2_channels):
    if is_spiral:
        module = di.spiral
    else:
        module = di.scalogram
    return module.build_graph(
                is_Z_supervision,
                n_bins_per_octave,
                n_octaves,
                X_width,
                conv1_channels,
                conv1_height,
                conv1_width,
                pool1_height,
                pool1_width,
                conv2_channels,
                conv2_height,
                conv2_width,
                pool2_height,
                pool2_width,
                drop1_proportion,
                dense1_channels,
                drop2_proportion,
                dense2_channels)

def substract_and_mask(args):
    return (args[0] - args[1]) * args[2]
