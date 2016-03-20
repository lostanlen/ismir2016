rastamat_path = fullfile('~', 'MATLAB', 'rastamat');
rwc_path = fullfile('~', 'datasets', 'rwc');
addpath(rastamat_path);

% MFCC settings
setting.arch = 'mfcc';
setting.numcep = 13;

% Parse RWC folder
file_metas = parse_rwc(rwc_path);
nBatches = length(unique([file_metas.batch_id]));

instrument_names = {'Clarinet', 'Flute', 'Trumpet', 'Piano', ...
    'Tenor Saxophone', 'Violin'};
batches = intersect( ...
    unique([file_metas(ismember({file_metas.instrument_name}, ...
    instrument_names)).batch_id]), ...
    unique([file_metas([file_metas.style_id]<3).batch_id]));

%% Compute MFCC features
for batch_id = 1:length(batches)
    batch = batches(batch_id);
    instrument = ...
        unique([file_metas([file_metas.batch_id]==batch).instrument_id]);
    pitches = ...
        unique([file_metas([file_metas.instrument_id]==instrument).pitch_id]);
    pitch_median = median(pitches);
    pitch_min = pitch_median - 16;
    pitch_max = pitch_median + 14;
    compute_batch(batch, file_metas, setting, pitch_min, pitch_max, rwc_path);
end

%% Load features
features = load_features(setting, batches, 'max')

%% Measure intra-class distances at fixed nuance, pitch, and style
percentiles = [0.1 0.25 0.5 0.75 0.9] * 100;
nPercentiles = length(percentiles);

nInstruments = length(instrument_names);
instrument_dists = cell(1, nInstruments);
instruments = zeros(1, nInstruments);
for instrument_id = 1:nInstruments
    instrument_name = instrument_names(instrument_id);
    instruments(instrument_id) = ...
        unique([file_metas(strcmp({file_metas.instrument_name}, ...
        instrument_name)).instrument_id]);
    instrument = instruments(instrument_id);
    in_cluster = [features([features.instrument_id]==instrument).data].';
    out_cluster = [features([features.instrument_id]~=instrument).data].';
    instrument_dists{instrument_id} = ...
        pdist2(in_cluster, out_cluster, 'euclidean').^2;
end
all_instrument_dists = [instrument_dists{:}];
all_instrument_dists = all_instrument_dists(:);
hist(all_instrument_dists, 1000);


%% Measure distances within instruments, with different pitch, nuance and style
note_dists = cell(1, nInstruments);
for instrument_id = 1:nInstruments
    instrument = instruments(instrument_id);
    instrument_features = features([features.instrument_id] == instrument);
    nNotes = length(instrument_features);
    pitches = [instrument_features.pitch_id];
    nuances = [instrument_features.nuance_id];
    styles = [instrument_features.style_id];
    a_distances = cell(1, nNotes);
    for a_index = 1:nNotes
        a = instrument_features(a_index);
        bs = instrument_features( ...
            (pitches ~= a.pitch_id) & ...
            (nuances ~= a.nuance_id) & ...
            (styles ~= a.style_id));
        a_distances{a_index} = pdist2(a.data.', [bs.data].').^2;
    end
    note_dists{instrument_id} = [a_distances{:}];
end


%% Measure distances within instrument, between notes of same pitch
pitch_dists = cell(1, nInstruments);
for instrument_id = 1:nInstruments
    instrument = instruments(instrument_id);
    instrument_features = features([features.instrument_id] == instrument);
    nNotes = length(instrument_features);
    pitches = [instrument_features.pitch_id];
    nuances = [instrument_features.nuance_id];
    styles = [instrument_features.style_id];
    a_distances = cell(1, nNotes);
    for a_index = 1:nNotes
        a = instrument_features(a_index);
        bs = instrument_features(pitches == a.pitch_id);
        a_distances{a_index} = pdist2(a.data.', [bs.data].').^2;
    end
    pitch_dists{instrument_id} = [a_distances{:}];
end

%% Same nuance
nuance_dists = cell(1, nInstruments);
for instrument_id = 1:nInstruments
    instrument = instruments(instrument_id);
    instrument_features = features([features.instrument_id] == instrument);
    nNotes = length(instrument_features);
    pitches = [instrument_features.pitch_id];
    nuances = [instrument_features.nuance_id];
    styles = [instrument_features.style_id];
    a_distances = cell(1, nNotes);
    for a_index = 1:nNotes
        a = instrument_features(a_index);
        bs = instrument_features(nuances == a.nuance_id);
        a_distances{a_index} = pdist2(a.data.', [bs.data].').^2;
    end
    nuance_dists{instrument_id} = [a_distances{:}];
end

%% Same style
style_dists = cell(1, nInstruments);
for instrument_id = 1:nInstruments
    instrument = instruments(instrument_id);
    instrument_features = features([features.instrument_id] == instrument);
    nNotes = length(instrument_features);
    pitches = [instrument_features.pitch_id];
    nuances = [instrument_features.nuance_id];
    styles = [instrument_features.style_id];
    a_distances = cell(1, nNotes);
    for a_index = 1:nNotes
        a = instrument_features(a_index);
        bs = instrument_features(styles == a.style_id);
        a_distances{a_index} = pdist2(a.data.', [bs.data].').^2;
    end
    style_dists{instrument_id} = [a_distances{:}];
end

%%
cellfun(@median, note_dists)
cellfun(@median, style_dists)
cellfun(@median, nuance_dists)
cellfun(@median, pitch_dists)