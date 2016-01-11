import collections
import DeepInstruments as di
import joblib
import medleydb
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
    u'Meaxic_TakeAStep_STEM_03.wav',  # overdubs
    u'TheSoSoGLos_Emergency_STEM_04.wav',  # overdubs
    # Female singer
    u'BrandonWebster_DontHearAThing_STEM_02.wav',  # has bleed
    # Male singer
    # Piano
    u'ClaraBerryAndWooldog_Boys_STEM_05.wav',  # has bleed
    u'MatthewEntwistle_TheArch_STEM_18.wav',  # has huge reverb
    # Trumpet
    u'MusicDelta_FusionJazz_STEM_06.wav'  # has sordina
    # Violin
    # (empty)

]

training_to_test = [
    # Clean electric guitar
    # (empty)
    # Distorted electric guitar
    # (empty)
    # Female singer
    u'HopAlong_SisterCities_STEM_07',  # to enrich test set
    u'LizNelson_Coldwar_STEM_02.wav',  # to avoid artist bias
    u'LizNelson_ImComingHome_STEM_01.wav',  # to avoid artist bias
    u'LizNelson_ImComingHome_STEM_04.wav',  # to avoid artist bias
    u'LizNelson_Rainfall_STEM_01.wav',  # to avoid artist bias
    # Male singer
    u'BigTroubles_Phantom_STEM_04.wav'  # to avoid artist bias
    u'HeladoNegro_MitadDelMundo_STEM_08.wav',  # to avoid artist bias
    u'StevenClark_Bounty_STEM_08.wav',  # to avoid artist bias
    # Piano
    # Trumpet
    u'MusicDelta_Beethoven_STEM_14.wav',  # to avoid song bias
    u'MusicDelta_ModalJazz_STEM_05.wav',  # to enrich test set
    # Violin
    u'JoelHelander_Definition_STEM_02.wav',
    u'JoelHelander_ExcessiveResistancetoChange_STEM_13.wav',
    u'JoelHelander_IntheAtticBedroom_STEM_01.wav'
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
    u'Creepoid_OldTree_STEM_09.wav',  # is a vocal FX track
    # Piano
    # Trumpet
    # (empty)
    # Violin
    # (empty)
]

cachedir = os.path.expanduser('~/joblib')
memory = joblib.Memory(cachedir=cachedir, verbose=0)


def confusion_matrix(Y_true, Y_predicted):
    y_true = np.argmax(Y_true, axis=1)
    y_predicted = np.argmax(Y_predicted, axis=1)
    n_classes = np.size(Y_true, 1)
    labels = range(n_classes)
    cm = sklearn.metrics.confusion_matrix(y_true, y_predicted, labels)
    cm = cm.astype('float64')
    return cm / cm.sum(axis=1)[:, np.newaxis]


def get_activation(stem):
    track_activations = np.vstack(stem.track.activations_data)[:, 1:]
    stem_id = int(stem.name[1:])
    return track_activations[:, stem_id - 1]


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


def test_accuracy(X_test, Y_test, batch_size, epoch_size, graph):
    labels = range(Y_test.shape[1])
    test_prediction = graph.predict({"X": X_test})
    Y_test_predicted = test_prediction["Y"]
    y_test_predicted = np.argmax(Y_test_predicted, axis=1)
    y_test_true = np.argmax(Y_test, axis=1)
    cm = sklearn.metrics.confusion_matrix(y_test_true, y_test_predicted,
                                          labels)
    cm = cm.astype(np.float64)
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    test_accuracies = np.diag(cm)
    mean_accuracy = np.mean(test_accuracies)
    std_accuracy = np.std(test_accuracies)
    print "test mean accuracy = ", mean_accuracy, " +/- ", std_accuracy
    return test_accuracies


def train_accuracy(X_train_list, Y_train_list,
                   batch_size, datagen, epoch_size, graph):
    labels = len(Y_train_list)
    dataflow = datagen.flow(X_train_list, Y_train_list,
        batch_size=batch_size,
        epoch_size=epoch_size)
    y_train_true = np.zeros(epoch_size, dtype=int)
    y_train_predicted = np.zeros(epoch_size, dtype=int)
    batch_id = 0
    for (X_batch, Y_batch) in dataflow:
        Y_batch_predicted = graph.predict_on_batch({"X": X_batch})
        Y_batch_predicted = np.hstack(Y_batch_predicted)
        y_batch_predicted = np.argmax(Y_batch_predicted, axis=1)
        batch_range = xrange(batch_id*batch_size, (batch_id+1)*batch_size)
        y_train_predicted[batch_range] = y_batch_predicted
        y_batch_true = np.argmax(Y_batch, axis=1)
        y_train_true[batch_range] = y_batch_true
        batch_id += 1
    cm = sklearn.metrics.confusion_matrix(y_train_true, y_train_predicted,
                                          labels)
    cm = cm.astype(np.float64)
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    train_accuracies = np.diag(cm)
    mean_accuracy = np.mean(train_accuracies)
    std_accuracy = np.std(train_accuracies)
    print "train accuracy = ", mean_accuracy, " +/- ", std_accuracy
    return train_accuracies


def split_stems(names,
                test_discarded,
                training_discarded,
                training_to_test,
                stems):
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
