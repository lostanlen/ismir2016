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
    'Tenor Saxophone', 'Piano', 'Violin'};
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

instruments = unique([features.instrument_id]);
nInstruments = length(instruments);
instrument_dists = cell(1, nInstruments);
for instrument_id = 1:nInstruments
    instrument = instruments(instrument_id);
    in_cluster = [features([features.instrument_id]==instrument).data].';
    out_cluster = [features([features.instrument_id]~=instrument).data].';
    instrument_dists{instrument_id} = ...
        pdist2(in_cluster, out_cluster, 'euclidean');
end
all_instrument_dists = [instrument_dists{:}];
all_instrument_dists = all_instrument_dists(:);
hist(all_instrument_dists, 1000);

%%
pitch_dists = cell(1, nInstruments);
all_pitch_dists = cell(1, nInstruments);
for instrument_id = 1:nInstruments
    instrument = instruments(instrument_id);
    instrument_features = features([features.instrument_id]==instrument);
    pitches = unique([instrument_features.pitch_id]);
    nPitches = length(pitches)
    pitch_dists{instrument_id} = cell(1, nPitches);
    for pitch_id = 1:nPitches
        pitch = pitches(pitch_id);
        in_cluster = ...
            [instrument_features([instrument_features.pitch_id]==pitch).data].';
        out_cluster = ...
            [instrument_features([instrument_features.pitch_id]~=pitch).data].';
        pitch_dists{instrument_id}{pitch_id} = ...
            pdist2(in_cluster, out_cluster, 'euclidean');
    end
    all_pitch_dists = [pitch_dists{instrument_id}{:}];
end