import DeepInstruments as di
import librosa
import medleydb
import numpy as np


def get_melody(stem):
    track_id = stem.track.id
    tracks = medleydb.sql.session().query(medleydb.sql.model.Track).all()
    track = [ track for track in tracks if track.id==track_id][0]
    melody_3rd_definition = track.melodies[2]
    if melody_3rd_definition.annotation_data:
        melodic_f0s = np.vstack(melody_3rd_definition.annotation_data)[:, 1:]
        if stem.rank:
            melody = melodic_f0s[:, stem.rank - 1]
        else:
            melody = np.zeros(melodic_f0s.shape[0])
    else:
        melody = np.zeros(len(track.activations_data) * 2048 / 256)
    return melody


def get_G(hop_length, mask_weight, n_bins_per_octave, n_octaves, stem):
    melody_f0s = di.symbolic.get_melody(stem)
    melody_annotation_hop = 256
    melody_downsampling = hop_length / melody_annotation_hop
    melody_range = xrange(0, len(melody_f0s), melody_downsampling)
    melody_f0s = melody_f0s[melody_range]
    gate = mask_weight * np.transpose((melody_f0s > 0.0).astype(np.float32))
    n_bins = n_bins_per_octave * n_octaves
    G = np.tile(gate, (n_bins, 1))
    return G


def get_Z(fmin, hop_length, n_bins_per_octave, n_octaves, stem):
    melody_f0s = di.symbolic.get_melody(stem)
    melody_annotation_hop = 256
    melody_downsampling = hop_length / melody_annotation_hop
    melody_range = xrange(0, len(melody_f0s), melody_downsampling)
    melody_f0s = melody_f0s[melody_range]
    midis = librosa.hz_to_midi(melody_f0s)
    midis[np.isinf(midis)] = 0.0
    track_activations = np.vstack(stem.track.activations_data)[:, 1:]
    stem_id = int(stem.name[1:]) - 1
    activations = track_activations[:, stem_id]
    activation_hop = 2048
    activation_upsampling = activation_hop / hop_length
    activations = activations.repeat(activation_upsampling)
    n_bins = n_bins_per_octave * n_octaves
    n_frames = len(activations)
    Z = np.zeros((n_bins, n_frames), np.float32)
    for frame_id in range(len(midis)):
        bin_id = int(midis[frame_id] - librosa.hz_to_midi(fmin)[0])
        if bin_id >= 0:
            Z[bin_id, frame_id] = activations[frame_id]
    return Z