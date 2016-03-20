function prefix = setting2prefix(setting)
prefix = ['rwc', setting.arch];
if strcmp(setting.arch, 'mfcc')
    prefix = ['rwcmfcc_numcep', num2str(setting.numcep, '%0.2d')];
else
    prefix = [prefix, '_Q', num2str(setting.Q, '%1.2d')];
    if isfield(setting, 'mu')
        mu_str = num2str(setting.mu, '%1.0e');
        prefix = [prefix, '_mu', mu_str];
    end
    if isfield(setting, 'F')
        F_str = num2str(setting.F, '%0.3d');
        prefix = [prefix, '_F', F_str];
    end
    if isfield(setting, 'B')
        B_str = num2str(setting.B, '%0.3d');
        prefix = [prefix, '_B', B_str];
    end
end
end
