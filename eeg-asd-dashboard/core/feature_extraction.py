"""
Python port of feature_extraction_v2.m
=======================================
Extracts 79 features per subject:
  A) Spectral Power (Welch PSD): 42 features
     - Absolute & relative band power (5 bands × 4 regions × 2) = 40
     - Frontal theta/alpha and theta/beta ratios = 2
  B) Functional Connectivity (Coherence): 32 features
     - 8 electrode pairs × 4 bands
  C) Sample Entropy: 5 features
     - 4 regions + global

References:
- Original MATLAB: feature_extraction_v2.m
- Welch PSD: scipy.signal.welch
- Coherence: scipy.signal.coherence
- Sample Entropy: antropy.sample_entropy
"""

import numpy as np
from scipy.signal import welch, coherence
from typing import Dict, List, Tuple
import antropy


class EEGFeatureExtractor:
    """
    Extracts multi-domain EEG features matching the thesis methodology.
    
    Feature domains:
    1. Spectral Power (Welch PSD)
    2. Functional Connectivity (Magnitude-Squared Coherence)
    3. Nonlinear Complexity (Sample Entropy)
    """
    
    def __init__(self, srate: float = 256):
        """
        Configure feature extraction constants and build feature names.

        The bands, regions, coherence pairs, and entropy settings are kept in
        one place so every extracted feature follows the thesis ordering.
        """
        self.srate = srate
        
        # Frequency bands (matching thesis Table 4)
        self.bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 15),
            'beta': (15, 30),
            'gamma': (30, 40)
        }
        
        # Region definitions (matching MATLAB regionDefs)
        self.regions = {
            'frontal': ['Fp1','AF7','AF3','F1','F3','F5','F7',
                       'Fpz','Fp2','AF8','Fz','F4','F6','F8'],
            'central': ['FC5','FC3','FC1','FC4','FC2','C3','C5','C4','C6'],
            'parietal': ['P1','P3','P5','P7','P9','Pz','P2','P4','P6',
                        'P8','P10','CP3','CP4','CP6'],
            'occipital': ['PO7','PO3','POz','PO8','PO4','O1','Oz','O2','Iz']
        }
        
        # Coherence pairs (matching MATLAB connPairs)
        self.coherence_pairs = [
            ('F3', 'F4'),    # frontal interhemispheric short-range
            ('C3', 'C4'),    # central interhemispheric short-range
            ('P5', 'P6'),    # parietal interhemispheric (P5-P6 in GWO) short-range
            ('TP7', 'TP8'),  # temporal interhemispheric long-range
            ('O1', 'O2'),    # occipital interhemispheric short-range
            ('Fz', 'Pz'),   # midline fronto-posterior long-range   
            ('F3', 'P5'),    # left fronto-posterior long-range
            ('F4', 'P4'),    # right fronto-posterior long-range
        ]
        
        # Coherence bands (4 bands, no gamma for connectivity)
        self.conn_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 15),
            'beta': (15, 30)
        }
        
        # Sample Entropy parameters (matching MATLAB)
        self.sampen_m = 2      # embedding dimension
        self.sampen_r = 0.2    # tolerance factor (r * std)
        
        # Build feature names
        self.feature_names = self._build_feature_names()
    
    def _build_feature_names(self) -> List[str]:
        """
        Build the ordered list of all 79 feature names.

        The order must match the numeric feature vector produced by extract()
        and the CSV/model artifacts used during inference.
        """
        names = []
        
        # A) Spectral: abs + rel per region per band (40 features)
        for region in self.regions:
            for band in self.bands:
                names.append(f'abs_{band}_{region}')
                names.append(f'rel_{band}_{region}')
        
        # Frontal ratios (2 features)
        names.append('theta_alpha_F')
        names.append('theta_beta_F')
        
        # B) Coherence (32 features: 8 pairs × 4 bands)
        pair_labels = ['F3_F4', 'C3_C4', 'P5_P6', 'TP7_TP8',
                       'O1_O2', 'Fz_Pz', 'F3_P5', 'F4_P4']
        for pair_label in pair_labels:
            for band in self.conn_bands:
                names.append(f'coh_{band}_{pair_label}')
        
        # C) Sample Entropy (5 features)
        for region in self.regions:
            names.append(f'sampen_{region}')
        names.append('sampen_global')
        
        return names
    
    def extract(self, epochs_data: np.ndarray, 
                channel_names: List[str]) -> np.ndarray:
        """
        Extract all 79 features from epoched EEG data.
        
        Parameters
        ----------
        epochs_data : np.ndarray
            Shape: [n_epochs, n_channels, n_samples]
        channel_names : list
            Channel labels matching the data
            
        Returns
        -------
        np.ndarray : Feature vector of length 79
        """
        n_epochs, n_channels, n_samples = epochs_data.shape
        
        # === A) SPECTRAL POWER (Welch PSD) ===
        # Compute PSD for each epoch and channel, then average over epochs
        win_len = min(self.srate, n_samples)  # 1-second window
        noverlap = win_len // 2
        
        # PSD: [n_epochs, n_channels, n_freqs]
        all_psd = []
        for ep in range(n_epochs):
            epoch_psd = []
            for ch in range(n_channels):
                sig = epochs_data[ep, ch, :]
                freqs, pxx = welch(sig, fs=self.srate, 
                                   nperseg=win_len, noverlap=noverlap)
                epoch_psd.append(pxx)
            all_psd.append(epoch_psd)
        
        all_psd = np.array(all_psd)  # [n_epochs, n_channels, n_freqs]
        mean_psd = np.mean(all_psd, axis=0)  # [n_channels, n_freqs]
        
        # Compute band powers per channel
        abs_power = {}  # {band: [n_channels]}
        for band_name, (f_low, f_high) in self.bands.items():
            band_mask = (freqs >= f_low) & (freqs <= f_high)
            if np.any(band_mask):
                abs_power[band_name] = np.trapz(
                    mean_psd[:, band_mask], freqs[band_mask], axis=1
                )
            else:
                abs_power[band_name] = np.zeros(n_channels)
        
        # Total power per channel
        total_power = sum(abs_power.values())
        total_power[total_power == 0] = np.nan
        
        # Relative power
        rel_power = {}
        for band_name in self.bands:
            rel_power[band_name] = abs_power[band_name] / total_power
        
        # Regional averages
        features = []
        for region_name, region_channels in self.regions.items():
            ch_indices = [i for i, ch in enumerate(channel_names) 
                         if ch in region_channels]
            
            for band_name in self.bands:
                if ch_indices:
                    abs_regional = np.nanmean(abs_power[band_name][ch_indices])
                    rel_regional = np.nanmean(rel_power[band_name][ch_indices])
                else:
                    abs_regional = np.nan
                    rel_regional = np.nan
                features.append(abs_regional)
                features.append(rel_regional)
        
        # Frontal theta/alpha and theta/beta ratios
        frontal_idx = [i for i, ch in enumerate(channel_names) 
                      if ch in self.regions['frontal']]
        if frontal_idx:
            theta_frontal = np.nanmean(rel_power['theta'][frontal_idx])
            alpha_frontal = np.nanmean(rel_power['alpha'][frontal_idx])
            beta_frontal = np.nanmean(rel_power['beta'][frontal_idx])
            
            theta_alpha_ratio = theta_frontal / alpha_frontal if alpha_frontal > 0 else np.nan
            theta_beta_ratio = theta_frontal / beta_frontal if beta_frontal > 0 else np.nan
        else:
            theta_alpha_ratio = np.nan
            theta_beta_ratio = np.nan
        
        features.append(theta_alpha_ratio)
        features.append(theta_beta_ratio)
        
        # === B) FUNCTIONAL CONNECTIVITY (Coherence) ===
        for ch1_name, ch2_name in self.coherence_pairs:
            idx1 = self._find_channel(ch1_name, channel_names)
            idx2 = self._find_channel(ch2_name, channel_names)
            
            if idx1 is not None and idx2 is not None:
                # Average coherence across epochs
                coh_epochs = []
                for ep in range(n_epochs):
                    f_coh, cxy = coherence(
                        epochs_data[ep, idx1, :],
                        epochs_data[ep, idx2, :],
                        fs=self.srate,
                        nperseg=win_len,
                        noverlap=noverlap
                    )
                    coh_epochs.append(cxy)
                
                mean_coh = np.mean(coh_epochs, axis=0)
                
                for band_name, (f_low, f_high) in self.conn_bands.items():
                    band_mask = (f_coh >= f_low) & (f_coh <= f_high)
                    if np.any(band_mask):
                        features.append(np.mean(mean_coh[band_mask]))
                    else:
                        features.append(np.nan)
            else:
                # Missing pair — append NaN for all 4 bands
                for _ in self.conn_bands:
                    features.append(np.nan)
        
        # === C) SAMPLE ENTROPY ===
        # Compute per channel, average across epochs first
        sampen_per_channel = np.zeros(n_channels)
        for ch in range(n_channels):
            ch_sampen = []
            for ep in range(n_epochs):
                sig = epochs_data[ep, ch, :]
                if np.std(sig) > 0:
                    try:
                        se = antropy.sample_entropy(
                            sig, order=self.sampen_m,
                            metric='chebyshev'
                        )
                        # antropy returns array; take last value
                        ch_sampen.append(se[-1] if hasattr(se, '__len__') else se)
                    except:
                        ch_sampen.append(np.nan)
                else:
                    ch_sampen.append(np.nan)
            sampen_per_channel[ch] = np.nanmean(ch_sampen)
        
        # Regional entropy
        for region_name, region_channels in self.regions.items():
            ch_indices = [i for i, ch in enumerate(channel_names) 
                         if ch in region_channels]
            if ch_indices:
                features.append(np.nanmean(sampen_per_channel[ch_indices]))
            else:
                features.append(np.nan)
        
        # Global entropy
        features.append(np.nanmean(sampen_per_channel))
        
        return np.array(features, dtype=np.float64)
    
    def _find_channel(self, name: str, channel_names: list):
        """
        Return the index for a channel name, ignoring capitalization.

        This helper allows feature extraction to work with slightly different
        channel-label casing across datasets.
        """
        for i, ch in enumerate(channel_names):
            if ch.lower() == name.lower():
                return i
        return None
    
    def extract_from_features_csv(self, row: np.ndarray) -> np.ndarray:
        """
        Return a precomputed feature row without recalculating anything.

        This is used when the dashboard reads feature values directly from the
        Sheffield CSV instead of processing raw EEG epochs.
        """
        return row


# GWO-Selected features (13 features from thesis Table 8)
GWO_SELECTED_FEATURES = [
    'rel_theta_frontal',
    'rel_delta_parietal',
    'rel_theta_parietal',
    'abs_theta_occipital',
    'abs_alpha_occipital',
    'theta_alpha_F',
    'coh_delta_C3_C4',
    'coh_alpha_C3_C4',
    'coh_theta_P5_P6',
    'coh_delta_Fz_Pz',
    'coh_beta_F4_P4',
    'sampen_central',
    'sampen_global'
]
