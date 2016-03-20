% MFCC settings
setting.numcep = 13;

% Parse RWC folder
file_metas = parse_rwc('~/datasets/rwc');
nBatches = length(unique([file_metas.batch_id]));
%% Compute MFCC features
for batch_id = 1:nBatches
    compute_batch(batch_id, file_metas, setting);
end

%% Measure intra-class distances at fixed nuance, pitch, and style
feature_range = 2:13;
percentiles = [0.1 0.25 0.5 0.75 0.9] * 100;
nPercentiles = length(percentiles);

all_data = [features.data];
all_data = all_data(feature_range, :);
all_pdists = pdist(all_data.').^2;
all_prctiles = prctile(all_pdists, percentiles);

instrument_ids = [features.instrument_id];
nInstruments = max(instrument_ids);
instrument_prctiles = zeros(nInstruments, nPercentiles);
style_prctiles = zeros(nInstruments, nPercentiles);
nuance_prctiles = zeros(nInstruments, nPercentiles);
pitch_prctiles = zeros(nInstruments, nPercentiles);

for instrument_id = 1:nInstruments
    instrument_mask = (instrument_ids==instrument_id);
    instrument_features = features(instrument_mask);
    % Instrument
    instrument_data = [instrument_features.data];
    instrument_data = instrument_data(feature_range, :);
    instrument_pdists = pdist(instrument_data.', 'euclidean').^2;
    % Style
    styles = unique([instrument_features.style_id]);
    nStyles = length(styles);
    style_distances = zeros(1, nStyles);
    for style_index = 1:nStyles
        style_id = styles(style_index);
        style_ids = [instrument_features.style_id];
        style_mask = (style_ids==style_id);
        style_features = instrument_features(style_mask);
        style_data = [style_features.data];
        style_data = style_data(feature_range, :);
        style_pdists = pdist(style_data.', 'euclidean').^2;
        style_distances(style_id) = mean(style_pdists);
    end
    % Pitch
    pitches = unique([instrument_features.pitch_id]);
    nPitches = length(pitches);
    pitch_distances = zeros(1, nPitches);
    for pitch_index = 1:nPitches
        pitch_id = pitches(pitch_index);
        pitch_ids = [instrument_features.pitch_id];
        pitch_mask = (pitch_ids==pitch_id);
        pitch_features = instrument_features(pitch_mask);
        pitch_data = [pitch_features.data];
        pitch_data = pitch_data(feature_range, :);
        pitch_pdists = pdist(pitch_data.', 'euclidean').^2;
        pitch_distances(pitch_id) = mean(pitch_pdists);
    end
    % Nuance
    nuances = unique([instrument_features.nuance_id]);
    nNuances = length(nuances);
    nuance_distances = zeros(1, nNuances);
    for nuance_id = 1:nNuances
        nuance_ids = [instrument_features.nuance_id];
        nuance_mask = (nuance_ids==nuance_id);
        nuance_features = instrument_features(nuance_mask);
        nuance_data = [nuance_features.data];
        nuance_data = nuance_data(feature_range, :);
        nuance_pdists = pdist(nuance_data.', 'euclidean').^2;
        nuance_distances(nuance_id) = mean(nuance_pdists);
    end
    instrument_prctiles(instrument_id, :) = ...
        prctile(instrument_pdists, percentiles);
    nuance_prctiles(instrument_id, :) = ...
        prctile(nuance_pdists, percentiles);
    style_prctiles(instrument_id, :) = ...
        prctile(style_pdists, percentiles);
    pitch_prctiles(instrument_id, :) = ...
        prctile(pitch_pdists, percentiles);
end
%%
required_instruments = {'Clarinet', 'Flute', 'Trumpet', 'Piano', ...
    'Tenor Saxophone', 'Piano', 'Violin'};

summary = struct();

for required_id = 1:length(required_instruments)
    required_instrument = required_instruments{required_id};
    booleans = cellfun(@(x) strcmp(required_instrument, x), ...
        {features.instrument_name});
    instrument_id = features(find(booleans, 1)).instrument_id;
    instrument_summary = struct();
    instrument_summary.instrument_prctiles = ...
        instrument_prctiles(instrument_id, :);
    instrument_summary.nuance_prctiles = ...
        nuance_prctiles(instrument_id, :);
    instrument_summary.style_prctiles = ...
        style_prctiles(instrument_id, :);
    instrument_summary.pitch_prctiles = ...
        pitch_prctiles(instrument_id, :);
    required_instrument_string = strsplit(required_instrument, ' ');
    required_instrument_string = required_instrument_string{1};
    summary.(required_instrument_string) = instrument_summary;
end

save('ismir2016_fig1_data', 'summary')

%%
