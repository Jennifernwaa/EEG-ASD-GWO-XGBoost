% =========================================================================
%  feature_extraction_1.m
%  EEG Feature Extraction — Sheffield ASD Dataset
%
%  Features extracted per subject:
%   A) Spectral Power (Welch PSD)
%      - Absolute & relative band power (delta/theta/alpha/beta/gamma)
%        per region (frontal/central/parietal/occipital)  → 40 features
%      - Frontal theta/alpha and theta/beta ratios          →  2 features
%
%   B) Functional Connectivity (mscohere)
%      - Mean coherence in 4 bands for 8 electrode pairs   → 32 features
%
%   C) Nonlinear Complexity (Sample Entropy)
%      - Regional average SampEn (4 regions) + global avg  →  5 features
%
%  Total: 79 features
%
%  NOTES:
%   - All channels are guaranteed present after preprocessing (interpolated).
%   - PSD uses Welch with a 1-second Hamming window and 50% overlap.
%   - Coherence uses the same Welch parameters via mscohere.
%   - SampEn uses m=2, r=0.2*std (standard parameters).
%   - Features are averaged across epochs before export.
% =========================================================================

%% ---- Configuration ----
inputDir  = '/Users/admin/Desktop/Sheffield/PROCESSED_V3';
outputCsv = '/Users/admin/Desktop/Sheffield/features_sheffield_noica_withp34.csv';

% Frequency bands
bands.names  = {'delta','theta','alpha','beta','gamma'};
bands.ranges = [1 4; 4 8; 8 15; 15 30; 30 40];

% Region definitions (10-20 labels)
regionDefs.frontal  = {'Fp1','AF7','AF3','F1','F3','F5','F7', ...
                        'Fpz','Fp2','AF8','Fz','F4','F6','F8'};
regionDefs.central  = {'FC5','FC3','FC1','FC4','FC2','C3','C5','C4','C6'};
regionDefs.parietal = {'P1','P3','P5','P7','P9','Pz','P2','P4','P6','P8','P10', ...
                        'CP3','CP4','CP6'};
regionDefs.occipital= {'PO7','PO3','POz','PO8','PO4','O1','Oz','O2','Iz'};

% Coherence pairs
connPairs = { ...
    'F3','F4';   % frontal interhemispheric
    'C3','C4';   % central interhemispheric
    'P5','P6';   % parietal interhemispheric
    'TP7','TP8'; % temporal interhemispheric
    'O1','O2';   % occipital interhemispheric
    'Fz','Pz';   % midline fronto-posterior
    'F3','P5';   % left fronto-posterior
    'F4','P4'};  % right fronto-posterior

connBandNames  = {'delta','theta','alpha','beta'};
connBandRanges = [1 4; 4 8; 8 15; 15 30];

% Sample Entropy parameters
sampEn_m       = 2;    % embedding dimension
sampEn_rFactor = 0.2;  % tolerance = rFactor * std(signal)

%% ---- File list ----
files = dir(fullfile(inputDir, '*.set'));
nSubs = numel(files);
if nSubs == 0
    error('No .set files found in %s', inputDir);
end
fprintf('Found %d preprocessed .set files.\n', nSubs);

%% ---- Build header ----
header       = {'Subject'};
regionNames  = fieldnames(regionDefs);
bandNames    = bands.names;
nBands       = numel(bandNames);
nRegions     = numel(regionNames);

% A) Spectral power
for r = 1:nRegions
    rn = regionNames{r};
    for b = 1:nBands
        bn = bandNames{b};
        header{end+1} = sprintf('abs_%s_%s', bn, rn);  %#ok<SAGROW>
        header{end+1} = sprintf('rel_%s_%s', bn, rn);  %#ok<SAGROW>
    end
end
% Frontal ratios
header{end+1} = 'theta_alpha_F';
header{end+1} = 'theta_beta_F';

% B) Coherence
for p = 1:size(connPairs,1)
    pairName = sprintf('%s_%s', connPairs{p,1}, connPairs{p,2});
    for cb = 1:numel(connBandNames)
        header{end+1} = sprintf('coh_%s_%s', connBandNames{cb}, pairName); %#ok<SAGROW>
    end
end

% C) Sample Entropy
for r = 1:nRegions
    header{end+1} = sprintf('sampen_%s', regionNames{r}); %#ok<SAGROW>
end
header{end+1} = 'sampen_global';

nFeat    = numel(header) - 1;  % exclude Subject column
featMat  = nan(nSubs, nFeat);
subNames = cell(nSubs, 1);

fprintf('Feature vector size: %d\n', nFeat);

%% ---- Main loop ----
for s = 1:nSubs
    fname = files(s).name;
    fpath = fullfile(inputDir, fname);
    fprintf('\n[%d/%d] %s\n', s, nSubs, fname);

    try
        %% Load
        EEG = pop_loadset('filename', fname, 'filepath', inputDir);
        EEG = eeg_checkset(EEG);

        data   = EEG.data;                  % ch × samples × epochs
        fs     = EEG.srate;
        [nCh, nSamp, nEp] = size(data);
        labels = {EEG.chanlocs.labels};
        fprintf('  %d ch | %d samples/epoch | %d epochs | fs=%.0f Hz\n', ...
            nCh, nSamp, nEp, fs);

        %% Welch parameters (1-second window, 50% overlap)
        winLen   = min(fs, nSamp);  % 256 samples = 1 s (or shorter if epoch < 1 s)
        win      = hamming(winLen);
        noverlap = floor(winLen / 2);
        freqRes  = 0.5;             % Hz resolution
        freqVec  = 0:freqRes:40;    % frequencies at which PSD is evaluated

        %% ===== A) Spectral features =====
        % absPow: nEp × nCh × nBands
        absPow = nan(nEp, nCh, nBands);

        for ch = 1:nCh
            for ep = 1:nEp
                sig = double(squeeze(data(ch,:,ep)));
                if all(sig == 0) || any(isnan(sig)); continue; end

                [Pxx, f] = pwelch(sig, win, noverlap, freqVec, fs);

                for b = 1:nBands
                    fr  = bands.ranges(b,:);
                    idx = f >= fr(1) & f <= fr(2);
                    if any(idx)
                        absPow(ep,ch,b) = trapz(f(idx), Pxx(idx));
                    end
                end
            end
        end

        % Average over epochs → nCh × nBands
        absPow_ch = squeeze(mean(absPow, 1, 'omitnan'));

        % Total broadband power per channel (sum across bands)
        totalPow_ch = sum(absPow_ch, 2);                     % nCh × 1
        totalPow_ch(totalPow_ch == 0) = NaN;                 % avoid /0

        % Relative power
        relPow_ch = absPow_ch ./ totalPow_ch;                % nCh × nBands

        % Regional averages
        regionBands_abs = nan(nRegions, nBands);
        regionBands_rel = nan(nRegions, nBands);

        for r = 1:nRegions
            rn      = regionNames{r};
            chanList= regionDefs.(rn);
            idxR    = find(ismember(labels, chanList));

            if isempty(idxR)
                warning('  No channels for region %s — row will be NaN.', rn);
                continue;
            end
            regionBands_abs(r,:) = mean(absPow_ch(idxR,:), 1, 'omitnan');
            regionBands_rel(r,:) = mean(relPow_ch(idxR,:), 1, 'omitnan');
        end

        % Frontal theta/alpha and theta/beta ratios
        iF     = find(strcmp(regionNames, 'frontal'));
        iTheta = find(strcmp(bandNames,  'theta'));
        iAlpha = find(strcmp(bandNames,  'alpha'));
        iBeta  = find(strcmp(bandNames,  'beta'));

        if ~isempty(iF) && ~isempty(iTheta) && ~isempty(iAlpha) && ~isempty(iBeta)
            thetaF = regionBands_rel(iF, iTheta);
            alphaF = regionBands_rel(iF, iAlpha);
            betaF  = regionBands_rel(iF, iBeta);
            thetaAlphaF = thetaF / alphaF;
            thetaBetaF  = thetaF / betaF;
        else
            thetaAlphaF = NaN;
            thetaBetaF  = NaN;
        end

        %% ===== B) Coherence features =====
        nPairs     = size(connPairs,1);
        nConnBands = size(connBandRanges,1);
        cohFeat    = nan(nPairs, nConnBands);

        for p = 1:nPairs
            ch1   = connPairs{p,1};
            ch2   = connPairs{p,2};
            idx1  = find(strcmpi(labels, ch1));
            idx2  = find(strcmpi(labels, ch2));

            if isempty(idx1) || isempty(idx2)
                fprintf('  Pair %s-%s missing — coherence = NaN.\n', ch1, ch2);
                continue;
            end

            coh_ep = nan(nEp, nConnBands);

            for ep = 1:nEp
                sig1 = double(squeeze(data(idx1,:,ep)));
                sig2 = double(squeeze(data(idx2,:,ep)));

                if all(sig1==0)||all(sig2==0)||any(isnan(sig1))||any(isnan(sig2))
                    continue;
                end

                [Cxy, fC] = mscohere(sig1, sig2, win, noverlap, freqVec, fs);

                for cb = 1:nConnBands
                    fr    = connBandRanges(cb,:);
                    idxF  = fC >= fr(1) & fC <= fr(2);
                    if any(idxF)
                        coh_ep(ep,cb) = mean(Cxy(idxF));
                    end
                end
            end

            cohFeat(p,:) = mean(coh_ep, 1, 'omitnan');
        end

        %% ===== C) Sample Entropy =====
        sampEn_ch_ep = nan(nCh, nEp);

        for ch = 1:nCh
            for ep = 1:nEp
                sig = double(squeeze(data(ch,:,ep)));
                if all(sig==0)||any(isnan(sig)); continue; end
                r_tol = sampEn_rFactor * std(sig);
                if r_tol == 0; continue; end
                sampEn_ch_ep(ch,ep) = SampEn_1D(sampEn_m, r_tol, sig);
            end
        end

        sampEn_ch = mean(sampEn_ch_ep, 2, 'omitnan');  % nCh × 1

        sampEn_region = nan(nRegions,1);
        for r = 1:nRegions
            rn      = regionNames{r};
            chanList= regionDefs.(rn);
            idxR    = find(ismember(labels, chanList));
            if isempty(idxR); continue; end
            sampEn_region(r) = mean(sampEn_ch(idxR), 'omitnan');
        end

        sampEn_global = mean(sampEn_ch, 'omitnan');

        %% ===== Assemble feature row =====
        featRow = nan(1, nFeat);
        k = 1;

        % A) Regional abs & rel power (region × band × 2)
        for r = 1:nRegions
            for b = 1:nBands
                featRow(k) = regionBands_abs(r,b); k = k+1;
                featRow(k) = regionBands_rel(r,b); k = k+1;
            end
        end

        % Frontal ratios
        featRow(k) = thetaAlphaF; k = k+1;
        featRow(k) = thetaBetaF;  k = k+1;

        % B) Coherence
        for p = 1:nPairs
            for cb = 1:nConnBands
                featRow(k) = cohFeat(p,cb); k = k+1;
            end
        end

        % C) Sample Entropy
        for r = 1:nRegions
            featRow(k) = sampEn_region(r); k = k+1;
        end
        featRow(k) = sampEn_global; k = k+1; %#ok<NASGU>

        %% Sanity check
        if k-1 ~= nFeat
            error('Feature count mismatch: expected %d, got %d', nFeat, k-1);
        end

        featMat(s,:) = featRow;
        subNames{s}  = fname;
        fprintf('  Done. NaN features: %d/%d\n', sum(isnan(featRow)), nFeat);

    catch ME
        fprintf('  ERROR: %s\n', ME.message);
        subNames{s} = fname;
    end
end

%% ---- Write CSV ----
T = array2table(featMat, 'VariableNames', header(2:end));
T.Subject = subNames;
T = movevars(T, 'Subject', 'Before', 1);

fprintf('\nWriting to %s ...\n', outputCsv);
writetable(T, outputCsv);
fprintf('Done. %d subjects × %d features.\n', nSubs, nFeat);

%% ============================================================
%  Helper: Sample Entropy (1D, Chebyshev distance)
% ============================================================
function se = SampEn_1D(m, r, x)
% SampEn_1D  Compute sample entropy of a 1-D time series.
%
%   m  : embedding dimension
%   r  : tolerance (absolute, typically 0.2*std(x))
%   x  : input vector
%
%   Reference: Richman & Moorman, Am J Physiol 2000.

x = x(:)';
N = length(x);

if N <= m+2
    se = NaN;
    return;
end

% Pre-allocate template matrices
Xm  = zeros(N-m,   m  );
Xm1 = zeros(N-m-1, m+1);

for ii = 1:(N-m)
    Xm(ii,:)  = x(ii:ii+m-1);
end
for ii = 1:(N-m-1)
    Xm1(ii,:) = x(ii:ii+m);
end

% Count template matches (excluding self-matches, consistent with SampEn)
count_m  = 0;
count_m1 = 0;

for ii = 1:size(Xm,1)-1
    dist_m  = max(abs(Xm(ii+1:end,:)  - Xm(ii,:)),  [], 2);
    count_m = count_m + sum(dist_m <= r);

    dist_m1  = max(abs(Xm1(ii+1:end,:) - Xm1(ii,:)), [], 2);
    count_m1 = count_m1 + sum(dist_m1 <= r);

end

if count_m == 0 || count_m1 == 0
    se = NaN;  % undefined (no matches found at this tolerance)
else
    se = -log(count_m1 / count_m);
end
end
