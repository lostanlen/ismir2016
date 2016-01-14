import collections
import DeepInstruments as di
import joblib
import math
import medleydb
import medleydb.sql
import numpy as np
import os
import sklearn

names = [u'clarinet',
         u'clean electric guitar',
         u'distorted electric guitar',
         u'female singer',
         u'male singer',
         u'piano',
         u'trumpet',
         u'violin']

training_discarded = [
    # Clean electric guitar
    u'CelestialShore_DieForUs_STEM_05.wav',  # overdubs and shoegaze effects
    # Distorted electric guitar
    # (empty)
    # Female singer
    # (empty)
    # Male singer
    u'MusicDelta_Disco_STEM_04.wav',  # falsetto
    # Piano
    u'MatthewEntwistle_TheArch_STEM_18.wav',  # has huge reverb
    # Trumpet
    u'MusicDelta_FusionJazz_STEM_06.wav',  # has sordina
    # Violin
    # (empty)

]

training_to_test = [
    # Clean electric guitar
    # (empty)
    # Distorted electric guitar
    # (empty)
    # Female singer
    u'HopAlong_SisterCities_STEM_07.wav',  # to enrich test set
    u'LizNelson_Coldwar_STEM_02.wav',  # to avoid artist bias
    u'LizNelson_ImComingHome_STEM_01.wav',  # to avoid artist bias
    u'LizNelson_ImComingHome_STEM_04.wav',  # to avoid artist bias
    u'LizNelson_Rainfall_STEM_01.wav',  # to avoid artist bias
    # Male singer
    u'BigTroubles_Phantom_STEM_04.wav',  # to avoid artist bias
    u'HeladoNegro_MitadDelMundo_STEM_08.wav',  # to avoid artist bias
    u'StevenClark_Bounty_STEM_08.wav',  # to avoid artist bias
    # Piano
    # (empty)
    # Trumpet
    u'MusicDelta_Beethoven_STEM_14.wav',  # to avoid song bias
    u'MusicDelta_ModalJazz_STEM_05.wav',  # to enrich test set
    # Violin
    # (empty)
]

test_discarded = [
    # Clarinet
    u'MusicDelta_Beethoven_STEM_09.wav',  # same song in training
    # Clean electric guitar
    u'AlexanderRoss_GoodbyeBolero_STEM_02.wav',  # same song in training
    u'CelestialShore_DieForUs_STEM_05.wav',  # has shoegaze effects
    u'AlexanderRoss_VelvetCurtain_STEM_10.wav',  # is actually a reverb track
    u'TheDistricts_Vermont_STEM_06.wav'  # same song in training
    # Distorted electric guitar
    u'AClassicEducation_NightOwl_STEM_03.wav',  # same song in training
    u'AClassicEducation_NightOwl_STEM_06.wav',  # same song in training
    u'AClassicEducation_NightOwl_STEM_07.wav'  # same song in training
    u'BigTroubles_Phantom_STEM_03.wav',  # same song in training
    u'BigTroubles_Phantom_STEM_03.wav',  # same song in training
    u'Creepoid_OldTree_STEM_05.wav',  # same song in training
    u'HopAlong_SisterCities_STEM_08.wav',  # same song in training
    u'MusicDelta_Britpop_STEM_03.wav',  # is arguably not distorted at all
    u'MusicDelta_Rockabilly_STEM_03.wav',  # is overdrive, not distortion
    u'MusicDelta_SpeedMetal_STEM_03.wav',  # same song in training
    u'MusicDelta_Zeppelin_STEM_03.wav',  # same song in training
    u'PurlingHiss_Lolita_STEM_04.wav',  # same song in training
    u'TheScarletBrand_LesFleursDuMal_STEM_05.wav',  # same song in training
    # Female singer
    u'ClaraBerryAndWooldog_AirTraffic_STEM_07.wav',  # to avoid artist bias
    u'LizNelson_ImComingHome_STEM_02.wav',  # has bleed
    # Male singer
    u'AClassicEducation_NightOwl_STEM_08.wav',  # to avoid song bias
    u'Creepoid_OldTree_STEM_09.wav',  # is a vocal FX track
    # Piano
    # (empty)
    # Trumpet
    # (empty)
    # Violin
    # (empty)
]

cachedir = os.path.expanduser('~/joblib')
memory = joblib.Memory(cachedir=cachedir, verbose=0)

class ScalogramGenerator(object):
    def __init__(self,
                 decision_length,
                 fmin,
                 hop_length,
                 n_bins_per_octave,
                 n_octaves,
                 training_stems):
        self.decision_length = float(decision_length)
        self.fmin = fmin
        self.hop_length = float(hop_length)
        self.n_bins_per_octave = n_bins_per_octave
        self.n_octaves = n_octaves
        X = []
        Y = []
        delayed_get_X = joblib.delayed(di.audio.cached_get_X)
        for class_stems in training_stems:
            X.append(joblib.Parallel(n_jobs=-1)(
                delayed_get_X(decision_length, fmin, hop_length,
                              n_bins_per_octave, n_octaves, stem)
                for stem in class_stems
            ))
            Y.append([di.singlelabel.get_Y(stem) for stem in class_stems])
        X_mat = np.hstack([X_file for X_class in X for X_file in X_class])
        self.X_mean = np.mean(X_mat)
        self.X_std = np.std(X_mat)
        for instrument_id in range(len(X)):
            X[instrument_id] = [(X_file-self.X_mean) / self.X_std
                                for X_file in X[instrument_id]]
        self.X = X
        self.Y = Y
        self.indices = di.singlelabel.get_indices(Y, decision_length)
        n_instruments = len(X)
        durations = []
        for instrument_id in range(n_instruments):
            file_lengths = map(float, map(len, self.indices[instrument_id]))
            durations.append(file_lengths / np.sum(file_lengths))
        self.durations = durations

    def flow(self, batch_size=512, epoch_size=4096):
        half_X_hop = int(0.5 * self.decision_length / self.hop_length)
        n_batches = int(math.ceil(float(epoch_size) / batch_size))
        n_bins = self.X[0][0].shape[0]
        n_instruments = self.Y[0][0].shape[0]
        X_batch_size = (batch_size, 1, n_bins, 2 * half_X_hop)
        X_batch = np.zeros(X_batch_size, np.float32)
        Y_batch_size = (batch_size, n_instruments)
        Y_batch = np.zeros(Y_batch_size, np.float32)
        y_epoch_size = (n_batches, batch_size)
        y_epoch = np.random.randint(0, n_instruments, size=y_epoch_size)
        for batch_id in range(n_batches):
            for sample_id in range(batch_size):
                instrument_id = y_epoch[batch_id, sample_id]
                n_files = len(self.indices[instrument_id])
                durations = self.durations[instrument_id]
                file_id = np.random.choice(n_files, p=durations)
                Y_id = np.random.choice(self.indices[instrument_id][file_id])
                X_id = int(Y_id * 2048.0 / self.hop_length)
                X_range = xrange(X_id-half_X_hop, X_id+half_X_hop)
                X_batch[sample_id, :, :] = \
                    self.X[instrument_id][file_id][:, X_range]
                Y_batch[sample_id, :] = \
                    self.Y[instrument_id][file_id][:, Y_id]
            yield X_batch, Y_batch

    def chunk(self, test_stems):
        X = []
        Y = []
        delayed_get_X = joblib.delayed(di.audio.cached_get_X)
        for class_stems in test_stems:
            X.append(joblib.Parallel(n_jobs=-1)(
                delayed_get_X(self.decision_length,
                              self.fmin,
                              self.hop_length,
                              self.n_bins_per_octave,
                              self.n_octaves,
                              stem)
                for stem in class_stems
            ))
            Y.append([di.singlelabel.get_Y(stem) for stem in class_stems])
        for instrument_id in range(len(X)):
            X[instrument_id] = [(X_file-self.X_mean) / self.X_std
                                for X_file in X[instrument_id]]
        X_chunks = []
        Y_chunks = []
        half_X_hop = int(0.5 * self.decision_length / self.hop_length)
        Y_hop = int(0.5 * float(self.decision_length) / 2048)
        indices = di.singlelabel.get_indices(Y, self.decision_length)
        for instrument_id in range(len(X)):
            for file_id in range(len(X[instrument_id])):
                Y_id = Y_hop - 1
                last_index = indices[instrument_id][file_id][-1]
                while Y_id < last_index:
                    Y_chunk = Y[instrument_id][file_id][:, Y_id]
                    if np.max(Y_chunk) > 0.5:
                        X_id = int(Y_id * 2048.0 / self.hop_length)
                        X_range = xrange(X_id-half_X_hop, X_id+half_X_hop)
                        X_chunk = X[instrument_id][file_id][:, X_range]
                        X_chunks.append(X_chunk)
                        Y_chunks.append(Y_chunk)
                    Y_id += Y_hop
        return X_chunks, Y_chunks


def get_indices(Y, decision_length):
    indices_classes = []
    activation_hop = 2048
    half_trimming_length = int(0.5 * (decision_length / activation_hop))
    for Y_class in Y:
        indices_files = []
        for activations in Y_class:
            activation = np.max(activations, axis=0)
            left_bound = half_trimming_length
            right_bound = len(activation) - half_trimming_length
            indices = np.where(np.greater_equal(activation, 0.5))[0]
            indices = indices[np.where(
                    np.greater(indices, left_bound) &
                    np.less(indices, right_bound))[0]]
            indices_files.append(indices)
        indices_classes.append(indices_files)
    return indices_classes


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


def get_Y(stem):
    track_activations = np.vstack(stem.track.activations_data)[:, 1:]
    stem_id = int(stem.name[1:])
    n_frames = track_activations.shape[0]
    n_instruments = len(di.singlelabel.names)
    activations = np.zeros((n_instruments, n_frames))
    instrument_id = di.singlelabel.names.index(stem.instrument.name)
    activations[instrument_id, :] = track_activations[:, stem_id - 1]
    return activations


@memory.cache
def melody_annotation_durations():
    session = medleydb.sql.session()
    stems = session.query(medleydb.sql.model.Stem).all()
    melodic_stems = \
        [stem for stem in stems if stem.rank and not stem.track.has_bleed]
    melodic_stem_names = [stem.instrument.name for stem in melodic_stems]
    unique_melodic_names = np.unique(melodic_stem_names)
    n_melodic_names = len(unique_melodic_names)
    n_melodic_stems = len(melodic_stems)
    melodic_counts = np.zeros(n_melodic_names)
    for stem_index in range(n_melodic_stems):
        name = melodic_stem_names[stem_index]
        stem = melodic_stems[stem_index]
        melody_3rd_definition = stem.track.melodies[2]
        f0 = np.vstack(melody_3rd_definition.annotation_data)[:, stem.rank]
        melodic_counts[unique_melodic_names==name] += (f0 > 0).sum()
    sorting = melodic_counts.argsort()
    sorted_melodic_names = unique_melodic_names[sorting]
    sorted_melodic_counts = melodic_counts[sorting]
    sorted_melodic_durations = sorted_melodic_counts * 256.0 / 44100
    counter = collections.Counter(melodic_stem_names)
    n_files_per_name = [ counter[name] for name in sorted_melodic_names ]
    tuples = (sorted_melodic_names,
              n_files_per_name,
              sorted_melodic_durations)
    return np.transpose(np.vstack(tuples))


def split_stems(names,
                test_discarded,
                training_discarded,
                training_to_test):
    session = medleydb.sql.session()
    stems = session.query(medleydb.sql.model.Stem).all()
    stems = [ stem for stem in stems if not stem.track.has_bleed]
    training_stems = []
    test_stems = []
    for name in names:
        instrument_stems = [stem for stem in stems
                            if stem.instrument.name == name]
        training_instrument_stems = []
        test_instrument_stems = []
        for stem in instrument_stems:
            stem_filename = os.path.split(stem.audio_path)[1]
            if stem.rank:
                if stem_filename in training_discarded:
                    pass
                elif stem_filename in training_to_test:
                    test_instrument_stems.append(stem)
                else:
                    training_instrument_stems.append(stem)
            else:
                if stem_filename in test_discarded:
                    pass
                else:
                    test_instrument_stems.append(stem)
        training_stems.append(training_instrument_stems)
        test_stems.append(test_instrument_stems)
    return test_stems, training_stems


def train_accuracy(batch_size, datagen, epoch_size, graph):
    n_batches = int(math.ceil(float(epoch_size) / batch_size))
    n_instruments = datagen.Y[0][0].shape[0]
    dataflow = datagen.flow(batch_size=batch_size, epoch_size=epoch_size)
    y_train_predicted = np.zeros((n_batches, batch_size), dtype=int)
    y_train_true = np.zeros((n_batches, batch_size), dtype=int)
    batch_id = 0
    for (X_batch, Y_batch) in dataflow:
        Y_batch_predicted = graph.predict_on_batch({"X": X_batch})
        Y_batch_predicted = np.hstack(Y_batch_predicted)
        y_batch_predicted = np.argmax(Y_batch_predicted, axis=1)
        y_train_predicted[batch_id, :] = y_batch_predicted
        y_batch_true = np.argmax(Y_batch, axis=1)
        y_train_true[batch_id, :] = y_batch_true
        batch_id += 1
    y_train_predicted = np.ndarray.flatten(y_train_predicted)
    y_train_true = np.ndarray.flatten(y_train_true)
    cm = sklearn.metrics.confusion_matrix(y_train_true, y_train_predicted,
                                          range(n_instruments))
    cm = cm.astype(np.float64)
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    train_accuracies = 100 * np.diag(cm)
    mean_accuracy = np.mean(train_accuracies)
    std_accuracy = np.std(train_accuracies)
    print "train accuracy = ", mean_accuracy, " +/- ", std_accuracy
    return train_accuracies
