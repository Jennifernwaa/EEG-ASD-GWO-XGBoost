"""
Python port of batch_process_sheffield.m
=========================================
Faithfully replicates the MATLAB preprocessing pipeline:
1. Resample to 256 Hz
2. Bandpass filter (1-40 Hz) + Notch (50 Hz)
3. Bad channel detection (correlation-based)
4. ICA artefact removal (via MNE)
5. Channel interpolation
6. Common Average Reference
7. Epoching (2-second windows)
8. Amplitude threshold rejection (±100 µV)

References:
- Original MATLAB: batch_process_sheffield_v3.m
- Uses MNE-Python for EEG processing
"""

import numpy as np
import mne
from scipy.signal import butter, filtfilt, iirnotch
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class EEGPreprocessor:
    """
    Replicates the Sheffield EEG preprocessing pipeline.
    
    Parameters match the MATLAB implementation:
    - target_srate: 256 Hz (resampled from 512 Hz)
    - bandpass: [1, 40] Hz FIR filter
    - notch_freq: 50 Hz (UK mains frequency)
    - epoch_duration: 2.0 seconds
    - amplitude_threshold: 100 µV
    """
    
    def __init__(self, 
                 target_srate: int = 256,
                 bandpass: Tuple[float, float] = (1.0, 40.0),
                 notch_freq: float = 50.0,
                 epoch_duration: float = 2.0,
                 amplitude_threshold: float = 100e-6):
        """
        Store preprocessing parameters and the target Sheffield montage.

        These settings mirror the MATLAB pipeline so extracted features stay
        comparable to the original training workflow.
        """
        
        self.target_srate = target_srate
        self.bandpass = bandpass
        self.notch_freq = notch_freq
        self.epoch_duration = epoch_duration
        self.amplitude_threshold = amplitude_threshold
        
        # Standard 64-channel montage (matching Sheffield dataset)
        self.target_channels = [
            'Fp1','Fpz','Fp2','AF7','AF3','AFz','AF4','AF8',
            'F7','F5','F3','F1','Fz','F2','F4','F6','F8',
            'FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8',
            'T7','C5','C3','C1','Cz','C2','C4','C6','T8',
            'TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8',
            'P9','P7','P5','P3','P1','Pz','P2','P4','P6','P8','P10',
            'PO7','PO3','POz','PO4','PO8',
            'O1','Oz','O2','Iz'
        ]
        
        self.processing_log = []
    
    def process(self, raw_data: np.ndarray, srate: float, 
                channel_names: list) -> Dict:
        """
        Full preprocessing pipeline.
        
        Parameters
        ----------
        raw_data : np.ndarray
            Raw EEG data [channels × samples]
        srate : float
            Original sampling rate
        channel_names : list
            Channel labels
            
        Returns
        -------
        dict : Contains processed epochs, metadata, and log
        """
        self.processing_log = []
        
        # Step 1: Create MNE Raw object
        info = mne.create_info(
            ch_names=channel_names,
            sfreq=srate,
            ch_types='eeg'
        )
        raw = mne.io.RawArray(raw_data, info)
        raw.set_montage('standard_1020', on_missing='ignore')
        self._log(f"Loaded: {len(channel_names)} channels, {srate} Hz")
        
        # Step 2: Resample
        if srate != self.target_srate:
            raw.resample(self.target_srate)
            self._log(f"Resampled: {srate} → {self.target_srate} Hz")
        
        # Step 3: Bandpass filter (1-40 Hz)
        raw.filter(
            l_freq=self.bandpass[0], 
            h_freq=self.bandpass[1],
            method='fir',
            fir_design='firwin'
        )
        self._log(f"Bandpass filtered: {self.bandpass[0]}-{self.bandpass[1]} Hz")
        
        # Step 4: Notch filter (50 Hz)
        raw.notch_filter(freqs=self.notch_freq)
        self._log(f"Notch filter applied: {self.notch_freq} Hz")
        
        # Step 5: Bad channel detection and interpolation
        raw_copy = raw.copy()
        n_before = len(raw.ch_names)
        raw.info['bads'] = self._detect_bad_channels(raw)
        if raw.info['bads']:
            raw.interpolate_bads(reset_bads=True)
            self._log(f"Interpolated {len(raw.info['bads'])} bad channels")
        
        # Step 6: ICA artefact removal
        ica = mne.preprocessing.ICA(
            n_components=0.95,
            method='infomax',
            fit_params=dict(extended=True),
            random_state=42
        )
        ica.fit(raw)
        
        # Automatic component rejection (mimics ICLabel thresholds)
        eog_indices, _ = ica.find_bads_eog(raw, threshold=0.7)
        muscle_indices, _ = ica.find_bads_muscle(raw, threshold=0.7)
        bad_ics = list(set(eog_indices + muscle_indices))
        ica.exclude = bad_ics
        raw = ica.apply(raw)
        self._log(f"ICA: removed {len(bad_ics)} artifact components")
        
        # Step 7: Re-reference to Common Average
        raw.set_eeg_reference('average')
        self._log("Re-referenced to Common Average (CAR)")
        
        # Step 8: Epoch into 2-second windows
        events = mne.make_fixed_length_events(
            raw, duration=self.epoch_duration
        )
        epochs = mne.Epochs(
            raw, events, 
            tmin=0, tmax=self.epoch_duration,
            baseline=None, preload=True
        )
        self._log(f"Epoched: {len(epochs)} epochs × {self.epoch_duration}s")
        
        # Step 9: Amplitude threshold rejection (±100 µV)
        n_before_reject = len(epochs)
        epochs.drop_bad(reject=dict(eeg=self.amplitude_threshold))
        n_rejected = n_before_reject - len(epochs)
        self._log(f"Threshold rejection: {n_rejected} epochs removed, "
                  f"{len(epochs)} retained")
        
        return {
            'epochs': epochs,
            'data': epochs.get_data(),  # [n_epochs, n_channels, n_samples]
            'srate': self.target_srate,
            'channel_names': epochs.ch_names,
            'n_epochs': len(epochs),
            'processing_log': self.processing_log
        }
    
    def _detect_bad_channels(self, raw) -> list:
        """
        Detect channels that correlate poorly with the rest of the montage.

        Channels below the mean-correlation threshold are marked for MNE
        interpolation before ICA and epoching continue.
        """
        data = raw.get_data()
        n_ch = data.shape[0]
        correlations = np.corrcoef(data)
        
        # Channel is bad if its mean correlation with others is < 0.8
        bad_channels = []
        for i in range(n_ch):
            other_corrs = np.delete(correlations[i], i)
            if np.nanmean(np.abs(other_corrs)) < 0.8:
                bad_channels.append(raw.ch_names[i])
        
        return bad_channels
    
    def _log(self, message: str):
        """
        Append a human-readable processing step to the pipeline log.

        The dashboard can surface this list to explain what happened during
        preprocessing.
        """
        self.processing_log.append(message)


def preprocess_simulated_eeg(eeg_data: np.ndarray, srate: float = 256) -> Dict:
    """
    Simplified preprocessing for simulated/dashboard EEG.

    Applies essential steps without MNE overhead for real-time feel.
    It returns epoched, cleaned data in the same shape expected by feature
    extraction: [n_epochs, n_channels, n_samples].
    """
    n_channels, n_samples = eeg_data.shape
    
    # Bandpass filter (1-40 Hz)
    nyq = srate / 2
    b, a = butter(4, [1/nyq, 40/nyq], btype='band')
    filtered = np.zeros_like(eeg_data)
    for ch in range(n_channels):
        filtered[ch] = filtfilt(b, a, eeg_data[ch])
    
    # Notch filter (50 Hz)
    b_notch, a_notch = iirnotch(50, 30, srate)
    for ch in range(n_channels):
        filtered[ch] = filtfilt(b_notch, a_notch, filtered[ch])
    
    # Common Average Reference
    avg = np.mean(filtered, axis=0)
    filtered = filtered - avg
    
    # Epoch into 2-second windows
    epoch_samples = int(2.0 * srate)
    n_epochs = n_samples // epoch_samples
    epochs = filtered[:, :n_epochs * epoch_samples].reshape(
        n_channels, n_epochs, epoch_samples
    ).transpose(1, 0, 2)  # [n_epochs, n_channels, n_samples]
    
    # Amplitude rejection (±100 µV)
    max_amp = np.max(np.abs(epochs), axis=(1, 2))
    good_mask = max_amp < 100e-6
    clean_epochs = epochs[good_mask]
    
    return {
        'data': clean_epochs,
        'srate': srate,
        'n_epochs': len(clean_epochs),
        'n_rejected': int(np.sum(~good_mask))
    }
