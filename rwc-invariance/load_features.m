function features = load_features(setting, summarization_str)
% Generate prefix string
prefix = setting2prefix(setting);
nBatches = 32;
batch_features = cell(nBatches,1);
% Load batches
for batch_id = 1:nBatches
    disp(['loading batch #', num2str(batch_id, '%0.2d')])
    batch_id_str = num2str(batch_id, '%1.2d');
    file_name = ['batch', batch_id_str];
    file_path = [prefix, '/', file_name];
    load(file_path);
    rwcbatch = summarize_batch(rwcbatch, summarization_str);
    batch_features{batch_id} = rwcbatch;
end
% Convert cell array to vector
features = [batch_features{:}];
end
