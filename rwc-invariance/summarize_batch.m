function summarized_batch = summarize_batch(batch, summarization_str)
nFiles_in_batch = length(batch);
summarized_batch = batch;
switch summarization_str
    case 'none'
    case 'mean'
        for file_id = 1:nFiles_in_batch
            summarized_batch(file_id).data = ...
                mean(batch(file_id).data(:,2:6), 2);
        end
    case 'max'
        for file_id = 1:nFiles_in_batch
            summarized_batch(file_id).data = max(batch(file_id).data, [], 2);
        end
end
end
