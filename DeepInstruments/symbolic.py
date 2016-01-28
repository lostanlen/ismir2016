import DeepInstruments as di
import librosa
import numpy as np


def get_melody(stem):
    melody_3rd_definition = stem.track.melodies[2]
    if melody_3rd_definition.annotation_data:
        melodic_f0s = np.vstack(melody_3rd_definition.annotation_data)[:, 1:]
        if stem.rank:
            melody = melodic_f0s[:, stem.rank - 1]
        else:
            melody = np.zeros(melodic_f0s.shape[0])
    else:
        melody = np.zeros(len(stem.track.activations_data))
    return melody


def get_G(hop_length, n_filters_per_octave, n_octaves, stem):
    f0s = di.symbolic.get_melody(stem)
    melody_annotation_hop = 256
    downsampling = hop_length / melody_annotation_hop
    downsampling_range = xrange(0, len(f0s), downsampling)
    f0s = f0s[downsampling_range]
    gate = (f0s > 0.0).astype(np.float32)
    n_bins = n_filters_per_octave * n_octaves
    G = np.tile(f0s, (n_bins, 1))
    return G