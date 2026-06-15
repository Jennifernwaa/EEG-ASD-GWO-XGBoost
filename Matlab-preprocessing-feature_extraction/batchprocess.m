% =========================================================================
%  batch_process_sheffield_DEMO.m
%  Modified version that saves intermediate .set files after each step
%  for examiner demonstration purposes.
%  Run this on ONE subject to generate the step-by-step outputs.
% =========================================================================

%% ---- Configuration -----
eeglab;

input_dir  = '/Users/admin/Desktop/Sheffield/RAW';
output_dir = '/Users/admin/Desktop/Sheffield/PROCESSED_V5';
demo_dir   = '/Users/admin/Desktop/Sheffield/DEMO_STEPS';  % NEW: intermediate saves

% Create demo directory
if ~exist(demo_dir, 'dir')
    mkdir(demo_dir);
end

loc_file = 'standard-10-5-cap385.elp';

target_channels = { ...
    'Fp1','Fpz','Fp2', ...
    'AF7','AF3','AFz','AF4','AF8', ...
    'F7','F5','F3','F1','Fz','F2','F4','F6','F8', ...
    'FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8', ...
    'T7','C5','C3','C1','Cz','C2','C4','C6','T8', ...
    'TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8', ...
    'P9','P7','P5','P3','P1','Pz','P2','P4','P6','P8','P10', ...
    'PO7','PO3','POz','PO4','PO8', ...
    'O1','Oz','O2','Iz'};

min_epochs = 10;

%% ---- Load standard locations ----
standard_locations = readlocs(loc_file);
std_labels = {standard_locations.labels};

target_idx = zeros(1, numel(target_channels));
n_found    = 0;
for k = 1:numel(target_channels)
    idx = find(strcmpi(std_labels, target_channels{k}), 1);
    if ~isempty(idx)
        n_found = n_found + 1;
        target_idx(n_found) = idx;
    end
end
target_idx      = target_idx(1:n_found);
target_chanlocs = standard_locations(target_idx);

%% ---- Process ONE demo subject ----
files = dir(fullfile(input_dir, '*.set'));
demo_file = files(1).name;  % Change index to pick a different subject

fprintf('\n========================================\n');
fprintf('DEMO: Processing %s with intermediate saves\n', demo_file);
fprintf('========================================\n');

%% STEP 0: Load raw
EEG = pop_loadset('filename', demo_file, 'filepath', input_dir);
EEG = eeg_checkset(EEG);
fprintf('\n[STEP 0 - RAW] %d channels, %.1f seconds, %.1f Hz\n', ...
    EEG.nbchan, EEG.xmax, EEG.srate);
pop_saveset(EEG, 'filename', 'step0_raw.set', 'filepath', demo_dir);

%% STEP 1: Resample to 256 Hz
if EEG.srate ~= 256
    EEG = pop_resample(EEG, 256);
end
fprintf('[STEP 1 - RESAMPLE] %d channels, %.1f seconds, %d Hz\n', ...
    EEG.nbchan, EEG.xmax, EEG.srate);
pop_saveset(EEG, 'filename', 'step1_resampled.set', 'filepath', demo_dir);

%% STEP 2: Bandpass filter 1-40 Hz
EEG = pop_eegfiltnew(EEG, 1, 40);
fprintf('[STEP 2 - BANDPASS] 1-40 Hz FIR filter applied\n');
pop_saveset(EEG, 'filename', 'step2_bandpass.set', 'filepath', demo_dir);

%% STEP 3: Notch filter 48-52 Hz
EEG = pop_eegfiltnew(EEG, 48, 52, [], 1);
fprintf('[STEP 3 - NOTCH] 48-52 Hz notch filter applied\n');
pop_saveset(EEG, 'filename', 'step3_notch.set', 'filepath', demo_dir);

%% STEP 4: Assign channel locations
EEG = pop_chanedit(EEG, 'lookup', loc_file);
EEG = eeg_checkset(EEG);
pre_clean_labels = {EEG.chanlocs.labels};
fprintf('[STEP 4 - LOCATIONS] Channel locations assigned (%d channels)\n', EEG.nbchan);
pop_saveset(EEG, 'filename', 'step4_locations.set', 'filepath', demo_dir);

%% STEP 5: Clean bad channels + ASR
EEG = clean_rawdata(EEG, 5, -1, 0.8, -1, 20, -1);
post_clean_labels = {EEG.chanlocs.labels};
removed_by_ASR = setdiff(pre_clean_labels, post_clean_labels);
fprintf('[STEP 5 - ASR] Channels removed: %d (%s)\n', ...
    numel(removed_by_ASR), strjoin(removed_by_ASR, ', '));
fprintf('  Remaining: %d channels\n', EEG.nbchan);
pop_saveset(EEG, 'filename', 'step5_asr_cleaned.set', 'filepath', demo_dir);

%% STEP 6: Average re-reference (pre-ICA)
EEG = pop_reref(EEG, []);
fprintf('[STEP 6 - REREF PRE-ICA] Average re-reference applied\n');
pop_saveset(EEG, 'filename', 'step6_reref_preICA.set', 'filepath', demo_dir);

%% STEP 7: ICA decomposition
EEG = pop_runica(EEG, 'extended', 1, 'interupt', 'off');
fprintf('[STEP 7 - ICA] ICA decomposition complete (%d components)\n', size(EEG.icaweights, 1));
pop_saveset(EEG, 'filename', 'step7_ica_decomposed.set', 'filepath', demo_dir);

%% STEP 8: ICLabel + artifact component rejection
EEG = pop_iclabel(EEG, 'default');
classProb = EEG.etc.ic_classification.ICLabel.classifications;
badIC = find( ...
    classProb(:,2) > 0.70 | ...
    classProb(:,3) > 0.70 | ...
    classProb(:,4) > 0.70 | ...
    classProb(:,5) > 0.70 | ...
    classProb(:,6) > 0.70);

fprintf('[STEP 8 - ICLABEL] %d artifact components identified and removed\n', numel(badIC));

% Log which components were removed and why
for ic = 1:numel(badIC)
    [~, maxClass] = max(classProb(badIC(ic), 2:6));
    classNames = {'Muscle','Eye','Heart','Line Noise','Chan Noise'};
    fprintf('  IC%d: %s (%.1f%%)\n', badIC(ic), classNames{maxClass}, ...
        classProb(badIC(ic), maxClass+1)*100);
end

if ~isempty(badIC)
    EEG = pop_subcomp(EEG, badIC, 0);
end
fprintf('  Channels after ICA rejection: %d\n', EEG.nbchan);
pop_saveset(EEG, 'filename', 'step8_ica_cleaned.set', 'filepath', demo_dir);

%% STEP 9: Interpolate missing channels
current_labels = {EEG.chanlocs.labels};
target_labels  = {target_chanlocs.labels};
missing_labels = setdiff(target_labels, current_labels);

fprintf('[STEP 9 - INTERPOLATION] %d channels to interpolate: %s\n', ...
    numel(missing_labels), strjoin(missing_labels, ', '));

if ~isempty(missing_labels)
    % Find indices of missing channels in target_chanlocs
    interp_idx = zeros(1, numel(missing_labels));
    n_interp = 0;
    for m = 1:numel(missing_labels)
        idx = find(strcmpi(target_labels, missing_labels{m}), 1);
        if ~isempty(idx)
            n_interp = n_interp + 1;
            interp_idx(n_interp) = idx;
        end
    end
    interp_idx = interp_idx(1:n_interp);
    
    % Single indexing operation — avoids dissimilar structures error
    interp_chanlocs = target_chanlocs(interp_idx);
    
    EEG = pop_interp(EEG, interp_chanlocs, 'spherical');
end

% Reorder channels to match target montage
[~, order] = ismember(target_labels, {EEG.chanlocs.labels});
valid_order = order(order > 0);
EEG = pop_select(EEG, 'channel', valid_order);
fprintf('  Final channel count: %d\n', EEG.nbchan);
pop_saveset(EEG, 'filename', 'step9_interpolated.set', 'filepath', demo_dir);

%% STEP 10: Final Common Average Reference
EEG = pop_reref(EEG, []);
fprintf('[STEP 10 - FINAL CAR] Common average reference applied\n');
pop_saveset(EEG, 'filename', 'step10_final_reref.set', 'filepath', demo_dir);

%% STEP 11: Epoch into 2-second windows
EEG = eeg_regepochs(EEG, 'recurrence', 2, 'limits', [0 2]);
fprintf('[STEP 11 - EPOCHING] %d epochs of 2 seconds created\n', EEG.trials);
pop_saveset(EEG, 'filename', 'step11_epoched.set', 'filepath', demo_dir);

%% STEP 12: Amplitude threshold rejection
nEpochs_before = EEG.trials;
[EEG, nRej] = pop_eegthresh(EEG, 1, 1:EEG.nbchan, ...
    -100, 100, EEG.xmin, EEG.xmax, 0, 1);
fprintf('[STEP 12 - THRESHOLD] %d epochs rejected, %d retained (from %d)\n', ...
    nRej, EEG.trials, nEpochs_before);
pop_saveset(EEG, 'filename', 'step12_final_cleaned.set', 'filepath', demo_dir);

%% ---- Summary ----
fprintf('\n========================================\n');
fprintf('DEMO COMPLETE\n');
fprintf('========================================\n');
fprintf('Files saved in: %s\n', demo_dir);
fprintf('  step0_raw.set           → Original recording\n');
fprintf('  step1_resampled.set     → After 512→256 Hz\n');
fprintf('  step2_bandpass.set      → After 1-40 Hz filter\n');
fprintf('  step3_notch.set         → After 50 Hz notch\n');
fprintf('  step4_locations.set     → After location assignment\n');
fprintf('  step5_asr_cleaned.set   → After bad channel removal + ASR\n');
fprintf('  step6_reref_preICA.set  → After pre-ICA average reference\n');
fprintf('  step7_ica_decomposed.set→ After ICA decomposition\n');
fprintf('  step8_ica_cleaned.set   → After ICLabel artifact removal\n');
fprintf('  step9_interpolated.set  → After channel interpolation\n');
fprintf('  step10_final_reref.set  → After final CAR\n');
fprintf('  step11_epoched.set      → After 2-sec epoching\n');
fprintf('  step12_final_cleaned.set→ After ±100µV rejection\n');
fprintf('========================================\n');