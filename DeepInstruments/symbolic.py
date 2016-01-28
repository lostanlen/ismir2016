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


def get_Z(hop_length, stem):
    f0s = di.symbolic.get_melody(stem)
    midis = librosa.hz_to_midi(f0s)
    midis[np.isinf(midis)] = 0
    melody_annotation_hop = 256
    downsampling = hop_length / melody_annotation_hop