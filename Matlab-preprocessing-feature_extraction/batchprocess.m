% =========================================================================
%  batch_process_sheffield.m
%  EEG Preprocessing Pipeline — Sheffield ASD Dataset
% =========================================================================

%% ---- Configuration -----
eeglab;

input_dir  = '/Users/admin/Desktop/Sheffield/RAW';
output_dir = '/Users/admin/Desktop/Sheffield/PROCESSED_V5';

% Path to the standard 10-5 electrode location file (shipped with EEGLAB)
loc_file = 'standard-10-5-cap385.elp';

% Target montage: channel labels you expect every subject to have.
% List ONLY channels actually present in your cap; adjust as needed.
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

% Minimum number of epochs to keep a subject
min_epochs = 10;

%% ---- Load standard locations (used for interpolation) ----
standard_locations = readlocs(loc_file);

% Build a lookup map: label -> chanloc struct
std_labels = {standard_locations.labels};

% Build target_chanlocs: chanloc structs for our target montage
% Collect matching indices first, then index in one shot to avoid
% the "dissimilar structures" error from incremental struct concatenation.
target_idx = zeros(1, numel(target_channels));
n_found    = 0;
for k = 1:numel(target_channels)
    idx = find(strcmpi(std_labels, target_channels{k}), 1);
    if ~isempty(idx)
        n_found = n_found + 1;
        target_idx(n_found) = idx;
    else
        warning('Channel %s not found in standard location file — skipping.', target_channels{k});
    end
end
target_idx      = target_idx(1:n_found);          % trim unused slots
target_chanlocs = standard_locations(target_idx); % single indexing op — fields always match
fprintf('Target montage: %d channels defined.\n', numel(target_chanlocs));

%% ---- Get file list ----
files = dir(fullfile(input_dir, '*.set'));
if isempty(files)
    error('No .set files found in %s', input_dir);
end
fprintf('Found %d .set files to process.\n\n', numel(files));

% Log table for QC
log_table = table('Size',[0 5], ...
    'VariableTypes',{'string','double','double','string','double'}, ...
    'VariableNames',{'File','nChan_raw','nChan_after_ICA','Interpolated','nEpochs'});

%% ---- Main loop ----
for i = 1:numel(files)

    fprintf('\n========================================\n');
    fprintf('Processing %d/%d: %s\n', i, numel(files), files(i).name);
    fprintf('========================================\n');

    try
        %% 1. Load
        EEG = pop_loadset('filename', files(i).name, 'filepath', input_dir);
        EEG = eeg_checkset(EEG);
        fprintf('  Loaded: %d ch, %.1f s, %.1f Hz\n', ...
            EEG.nbchan, EEG.xmax, EEG.srate);

        %% 2. Resample
        if EEG.srate ~= 256
            EEG = pop_resample(EEG, 256);
        end

        %% 3. Bandpass 1–40 Hz
        EEG = pop_eegfiltnew(EEG, 1, 40);

        %% 4. Notch filter at 50 Hz (line noise)
        EEG = pop_eegfiltnew(EEG, 48, 52, [], 1);

        %% 5. Assign standard locations to any channel missing coordinates
        %     (critical — clean_rawdata & pop_interp need valid coordinates)
        EEG = pop_chanedit(EEG, 'lookup', loc_file);
        EEG = eeg_checkset(EEG);

        % Record which channels are present before cleaning
        pre_clean_labels = {EEG.chanlocs.labels};

        %% 6. Clean bad channels + ASR
        %     flat_crit=5, window_crit=-1 (off), corr_thresh=0.8,
        %     noise_crit=-1, high_pass=[-1] (already filtered),
        %     burst_crit=20 (µV), window_reject=-1 (off, we epoch later)
        EEG = clean_rawdata(EEG, 5, -1, 0.8, -1, 20, -1);
        post_clean_labels = {EEG.chanlocs.labels};
        removed_by_ASR = setdiff(pre_clean_labels, post_clean_labels);
        if ~isempty(removed_by_ASR)
            fprintf('  Channels removed by clean_rawdata: %s\n', ...
                strjoin(removed_by_ASR, ', '));
        end

        %% 7. Average re-reference (before ICA)
        EEG = pop_reref(EEG, []);
       
        %% 8. ICA (extended infomax)
        EEG = pop_runica(EEG, 'extended', 1, 'interupt', 'off');

        %% 9. ICLabel — remove non-brain components (threshold 0.70)
        EEG = pop_iclabel(EEG, 'default');

        % Classes: 1=Brain, 2=Muscle, 3=Eye, 4=Heart,
        %          5=Line Noise, 6=Chan Noise, 7=Other
        classProb = EEG.etc.ic_classification.ICLabel.classifications;
        badIC = find( ...
            classProb(:,2) > 0.70 | ...   % Muscle
            classProb(:,3) > 0.70 | ...   % Eye
            classProb(:,4) > 0.70 | ...   % Heart
            classProb(:,5) > 0.70 | ...   % Line noise
            classProb(:,6) > 0.70);       % Channel noise
        % Note: 'Other' (col 7) intentionally excluded to avoid over-rejection

        fprintf('  Removing %d ICA components.\n', numel(badIC));
        if ~isempty(badIC)
            EEG = pop_subcomp(EEG, badIC, 0);
        end

        % %% 10. Interpolate ALL channels in target montage that are missing
        % %      This covers BOTH channels removed by clean_rawdata AND
        % %      channels that were never recorded for this subject.
        current_labels = {EEG.chanlocs.labels};
        target_labels  = {target_chanlocs.labels};
        missing_labels = setdiff(target_labels, current_labels);

        if ~isempty(missing_labels)
            fprintf('  Interpolating %d missing channel(s): %s\n', ...
                numel(missing_labels), strjoin(missing_labels, ', '));

            % Build chanloc structs for the missing channels only
            interp_chanlocs = struct([]);
            for m = 1:numel(missing_labels)
                idx = find(strcmpi(target_labels, missing_labels{m}), 1);
                if ~isempty(idx)
                    interp_chanlocs(end+1) = target_chanlocs(idx); %#ok<AGROW>
                end
            end

            % pop_interp(EEG, badchanlocs, method)
            % We pass the chanlocs of CHANNELS TO ADD (not the full set)
            EEG = pop_interp(EEG, interp_chanlocs, 'spherical');
        else
            fprintf('  No channels to interpolate.\n');
        end
        
        % Reorder channels to match target montage order
        [~, order] = ismember(target_labels, {EEG.chanlocs.labels});
        valid_order = order(order > 0);
        EEG = pop_select(EEG, 'channel', valid_order);

        %% 11. Final CAR after interpolation
        EEG = pop_reref(EEG, []);

        %% 12. Epoch into 2-second windows (non overlapping)
        EEG = eeg_regepochs(EEG, 'recurrence', 2, 'limits', [0 2]);

        %% 13. Amplitude threshold rejection (±100 µV)
        [EEG, nRej] = pop_eegthresh(EEG, 1, 1:EEG.nbchan, ...
            -100, 100, EEG.xmin, EEG.xmax, 0, 1);
        fprintf('  Epochs rejected by threshold: %d\n', nRej);
        fprintf('  Epochs retained: %d\n', EEG.trials);

        if EEG.trials < min_epochs
            warning('  Subject %s has only %d epochs — consider excluding.', ...
                files(i).name, EEG.trials);
        end

        %% 14. Save
        out_name = [files(i).name(1:end-4), '_cleaned.set'];
        EEG = pop_saveset(EEG, 'filename', out_name, 'filepath', output_dir);
        fprintf('  Saved: %s\n', out_name);

        %% Log
        log_table(end+1,:) = { ...
            string(files(i).name), ...
            numel(pre_clean_labels), ...
            EEG.nbchan, ...
            string(strjoin(missing_labels, '; ')), ...
            EEG.trials};

    catch ME
        fprintf('  ERROR processing %s:\n  %s\n', files(i).name, ME.message);
        log_table(end+1,:) = { ...
            string(files(i).name), NaN, NaN, string('ERROR'), NaN};
    end
end

%% ---- Save QC log ----
log_path = fullfile(output_dir, 'preprocessing_log.csv');
writetable(log_table, log_path);
fprintf('\n\nPreprocessing complete.\nQC log saved to: %s\n', log_path);
