import DeepInstruments as di
import joblib
import librosa
import numpy as np
import os


def chunk_test_set(X_test_list, Y_test_list, hop_duration, sr):
    pass


def get_XY(file_paths,
           instrument_list, decision_duration, fmin, hop_duration,
           n_bins_per_octave, n_octaves, sr):
    # Run perceptual CQT in parallel with joblib
    # n_jobs = -1 means that all CPUs are used
    memory = joblib.Memory(cachedir=os.path.expanduser('~/joblib'))
    cached_cqt = memory.cache(di.audio.perceptual_cqt, verbose=0)
    X_files = joblib.Parallel(n_jobs=-1, verbose=0)(
            joblib.delayed(cached_cqt)(file_path,
                                       decision_duration,
                                       fmin,
                                       hop_duration,
                                       n_bins_per_octave,
                                       n_octaves,
                                       sr) for file_path in file_paths)
    # Gather file CQTs X according to each class
    # Generate one-hot vectors Y for each class
    n_files = len(file_paths)
    file_range = range(n_files)
    n_instruments = len(instrument_list)
    instr_range = range(n_instruments)
    Y_files = \
        [di.symbolic.get_instrument(p, instrument_list) for p in file_paths]
    X_list = []
    Y_list = []
    for i in instr_range:
        X_instrument = [X_files[f] for f in file_range if Y_files[f] == i]
        Y_instrument = np.zeros((1, n_instruments), dtype=np.float32)
        Y_instrument[0, i] = 1.0
        X_list.append(np.hstack(X_instrument))
        Y_list.append(Y_instrument)
    return (X_list, Y_list)


def get_activations(names, track):
    instrument_matches = di.wrangling.instrument_stems(names, track)
    stem_activations = track.activations_data
    stem_activations = np.vstack(stem_activations)[:, 1:]
    n_frames = stem_activations.shape[0]
    n_instruments = len(instrument_matches)
    activations = np.zeros((n_frames, n_instruments), dtype=np.float32)
    for instrument_index in range(n_instruments):
        stems = instrument_matches[instrument_index]
        if stems:
            instrument_stem_activations = stem_activations[:, stems]
            activations[:, instrument_index] = \
                np.max(instrument_stem_activations, axis=1)
    return activations


def get_pianorolls(fmin, melodic_names, n_bins_per_octave, n_octaves, track):
    # get melodic activations
    activation_hop_length = 2048.0
    melodic_activations = di.wrangling.get_activations(melodic_names, track)

    # get melodic f0s
    melody_3rd_definition = track.melodies[2]
    melodic_f0s = np.vstack(melody_3rd_definition.annotation_data)[:, 1:]
    melody_hop_length = 256.0
    downsampling_factor = int(activation_hop_length / melody_hop_length)
    n_melody_samples = melodic_f0s.shape[0]
    downsampling_range = range(0, n_melody_samples, downsampling_factor)
    melodic_f0s = melodic_f0s[downsampling_range, :]

    # converts f0s to MIDI pitches and to CQT bin indices
    # After this conversion, silent frames correspond to the value -inf
    melodic_pitches = librosa.hz_to_midi(melodic_f0s)
    melodic_bins = np.round(melodic_pitches) - librosa.hz_to_midi(fmin)

    # Initialize piano-rolls as a tensor of zeroes
    n_melodic_instruments = len(melodic_names)
    n_bins = n_bins_per_octave * n_octaves
    n_frames, n_melodies = melodic_bins.shape
    pianorolls_shape = (n_melodic_instruments, n_bins, n_frames)
    pianorolls = np.zeros(pianorolls_shape, dtype=np.float32)

    # Find ranks of stems in melody annotation
    stems = track.stems.all()
    ranks = [ stem.rank for stem in stems ]

    # For each annotated melody
    for melody_index in range(n_melodies):
        # Get the name of the corresponding instrument
        name = stems[ranks.index(melody_index+1)].instrument.name
        # Match this name to the list of melodic instruments
        name_index = melodic_names.index(name)
        # Write its melody activations at the right bins, frame by frame
        for frame_index in range(n_frames):
            melodic_bin = melodic_bins[frame_index, melody_index]
            if not np.isinf(melodic_bin):
                activation = melodic_activations[frame_index, name_index]
                pianorolls[name_index, melodic_bin, frame_index] = activation

    # Return all piano-rolls as a Numpy tensor
    return pianorolls


def instrument_stems(instrument_names, track):
    stem_names = [s.instrument.name for s in track.stems]
    n_instruments = len(instrument_names)
    n_stems = len(stem_names)
    instrument_matches = []
    for instrument_index in range(n_instruments):
        instrument_name = instrument_names[instrument_index]
        instrument_match = []
        for stem_index in range(n_stems):
            stem_name = stem_names[stem_index]
            if stem_name == instrument_name:
                instrument_match.append(stem_index)
        instrument_matches.append(instrument_match)
    return instrument_matches