function rwcbatch = ...
    compute_batch(batch_id, file_metas, setting, pitch_min, pitch_max, rwc_path)
%% Filter folder according to specified batch
rwcbatch = file_metas([file_metas.batch_id] == batch_id);

% Narrow the pitch range to pitches that have 3 nuances
p_pitches = [rwcbatch([rwcbatch.nuance_id] == 1).pitch_id];
mf_pitches = [rwcbatch([rwcbatch.nuance_id] == 2).pitch_id];
f_pitches = [rwcbatch([rwcbatch.nuance_id] == 3).pitch_id];
pitches = intersect(intersect(p_pitches, mf_pitches), f_pitches);

% Assert that pitches are along a chromatic scale
assert(isequal(pitches, (min(pitches):max(pitches))))
rwcbatch = rwcbatch([rwcbatch.pitch_id] >= pitch_min);
rwcbatch = rwcbatch([rwcbatch.pitch_id] <= pitch_max);

% Initialize fields signal, data, setting, and S
[rwcbatch.signal] = deal(0);
[rwcbatch.data] = deal(0);
[rwcbatch.S] = deal(0);
[rwcbatch.setting] = deal(setting);
nFiles = length(rwcbatch);

%% Measure elapsed time with tic() and toc()
tic();
numcep = setting.numcep;
parfor file_index = 1:nFiles
    file_meta = rwcbatch(file_index);
    subfolder = file_meta.subfolder;
    wavfile_name = file_meta.wavfile_name;
    file_path = [rwc_path, '/', subfolder, '/', wavfile_name];
    [signal, sample_rate] = audioread(file_path);
    mfcc = melfcc(signal, sample_rate , ...
        'wintime', 0.016, ...
        'lifterexp', 0, ...
        'minfreq', 133.33, ...
        'maxfreq', 6855.6, ...
        'sumpower', 0, ...
        'numcep', numcep);
    data = mfcc;
    rwcbatch(file_index).signal = signal;
    rwcbatch(file_index).data = data;
end

elapsed = toc();
elapsed_str = num2str(elapsed, '%2.0f');

% Get host name
pcinfo = java.net.InetAddress.getLocalHost();
host = pcinfo.getHostName(); % class is java.lang.String
host = char(host); % convert to MATLAB char array

% Get date
date = datestr(now());

% Save
batch_id_str = num2str(batch_id, '%1.2d');
prefix = ['rwcmfcc_numcep', num2str(setting.numcep, '%0.2d')];
savefile_name = ['batch', batch_id_str];
if ~exist(prefix,'dir')
    mkdir(prefix);
end
savefile_path = [prefix, '/', savefile_name];
save(savefile_path, 'rwcbatch', 'setting', 'host', 'elapsed', 'date');

% Print termination message
disp('--------------------------------------------------------------------------------');
disp(['Finished batch ', batch_id_str, ' on host ', host, ...
    ' at ', date,' with settings:']);
disp(setting);
disp(['Elapsed time is ', elapsed_str ' seconds.']);
end
