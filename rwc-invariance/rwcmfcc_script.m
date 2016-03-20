%% Settings
setting.arch = 'mfcc';

%% Parse RWC folder
file_metas = parse_rwc('~/datasets/rwc');
nBatches = length(unique([file_metas.batch_id]));

%% This loop in computed in the cluster
numcep = 40;
newsetting = setting;
for numcep = numceps
    setting.numcep = numcep;
     for batch_id = 1:nBatches
         compute_batch(batch_id, file_metas, setting);
     end
    features = load_features(setting, 'max');
    measure_icc(features)
    prefix = setting2prefix(newsetting);
    save([prefix, '/', prefix, '_icc'], 'icc');
end

%% Load features and max-pool across time
features = load_features(setting, 'max');

%% Measure distances
summary = compute_average_distances(setting, features, 'euclidean');
summary = compute_average_distances(setting, features, 'cosine');

%% Compute tSNE
addpath(genpath('~/MATLAB/tSNE'));
data = [features.data].';
labels = [features.instrument_id];
dims_in = size(data, 2);
dims_out = 2;
y = tsne(data, labels, dims_out, dims_in);
save([prefix,'/',prefix,'_tsne'], y);
%%
labels = [features.nuance_id]
scatter(y(:,1),y(:,2), 9, labels, 'filled')