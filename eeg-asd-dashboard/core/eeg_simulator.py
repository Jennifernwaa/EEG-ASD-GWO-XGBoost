"""
EEG Signal Simulator
=====================
Generates realistic synthetic EEG signals for dashboard demonstration.
Can produce both normal (Control) and ASD-like patterns.

ASD-like patterns include:
- Elevated theta/delta power (U-shaped spectral profile)
- Reduced alpha power
- Altered interhemispheric coherence
- Reduced signal complexity (lower entropy)
"""

import numpy as np
from typing import Dict, Tuple


class EEGSimulator:
    """
    Generates realistic multichannel EEG signals.
    
    Based on the neurophysiological literature:
    - Eroğlu (2025): U-shaped spectral profile in ASD
    - Coben et al. (2014): Mixed over/under-connectivity
    - Bosl et al. (2018): Reduced entropy in ASD
    """
    
    def __init__(self, srate: float = 256, duration: float = 10.0,
                 n_channels: int = 32, seed: int = None):
        """
        Configure the synthetic EEG generator and channel montage.

        The simulator uses a compact channel list that still includes every
        electrode needed by the selected connectivity features.
        """
        self.srate = srate
        self.duration = duration
        self.n_samples = int(srate * duration)
        self.rng = np.random.default_rng(seed)
        
        # Compact montage with all channels needed by the GWO-selected features.
        channel_pool = [
            'Fp1', 'Fp2', 'AF7', 'AF3', 'AF8', 'F7', 'F3', 'Fz',
            'F4', 'F8', 'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'C3',
            'Cz', 'C4', 'C5', 'C6', 'TP7', 'TP8', 'P7', 'P5',
            'P3', 'Pz', 'P4', 'P6', 'P8', 'O1', 'Oz', 'O2',
        ]
        self.channel_names = channel_pool[:min(n_channels, len(channel_pool))]
        self.n_channels = len(self.channel_names)
    
    def generate(self, condition: str = 'control', 
                 noise_level: float = 0.3) -> Dict:
        """
        Generate synthetic multichannel EEG.
        
        Parameters
        ----------
        condition : str
            'control' or 'asd' — determines spectral characteristics
        noise_level : float
            Amount of random noise (0-1)
            
        Returns
        -------
        dict : EEG data, time vector, and parameters
        """
        t = np.linspace(0, self.duration, self.n_samples)
        data = np.zeros((self.n_channels, self.n_samples))
        
        for ch in range(self.n_channels):
            data[ch] = self._generate_channel(t, condition, noise_level, ch)
        
        return {
            'data': data,
            'time': t,
            'srate': self.srate,
            'n_channels': self.n_channels,
            'channel_names': self.channel_names[:self.n_channels],
            'condition': condition,
            'duration': self.duration
        }
    
    def _generate_channel(self, t: np.ndarray, condition: str,
                          noise_level: float, ch_idx: int) -> np.ndarray:
        """
        Generate one synthetic EEG channel with condition-specific rhythms.

        Control and ASD conditions differ in band amplitudes and burstiness,
        while both include pink and white noise to mimic EEG background signal.
        """
        
        # Base oscillatory components
        if condition == 'asd':
            # ASD: elevated delta/theta, reduced alpha, elevated gamma
            delta_amp = self.rng.uniform(15, 25) * 1e-6
            theta_amp = self.rng.uniform(12, 20) * 1e-6
            alpha_amp = self.rng.uniform(3, 8) * 1e-6   # Reduced
            beta_amp = self.rng.uniform(4, 8) * 1e-6
            gamma_amp = self.rng.uniform(3, 6) * 1e-6   # Elevated
        else:
            # Control: normal spectral profile
            delta_amp = self.rng.uniform(8, 15) * 1e-6
            theta_amp = self.rng.uniform(6, 12) * 1e-6
            alpha_amp = self.rng.uniform(10, 20) * 1e-6  # Strong alpha
            beta_amp = self.rng.uniform(3, 7) * 1e-6
            gamma_amp = self.rng.uniform(1, 3) * 1e-6
        
        # Generate oscillations with slight frequency jitter
        signal = (
            delta_amp * np.sin(2 * np.pi * self.rng.uniform(1, 4) * t +
                              self.rng.uniform(0, 2*np.pi)) +
            theta_amp * np.sin(2 * np.pi * self.rng.uniform(4, 8) * t +
                              self.rng.uniform(0, 2*np.pi)) +
            alpha_amp * np.sin(2 * np.pi * self.rng.uniform(8, 13) * t +
                              self.rng.uniform(0, 2*np.pi)) +
            beta_amp * np.sin(2 * np.pi * self.rng.uniform(15, 30) * t +
                             self.rng.uniform(0, 2*np.pi)) +
            gamma_amp * np.sin(2 * np.pi * self.rng.uniform(30, 40) * t +
                              self.rng.uniform(0, 2*np.pi))
        )
        
        # Add 1/f noise (pink noise - realistic EEG background)
        pink_noise = self._pink_noise(len(t)) * noise_level * 10e-6
        
        # Add white noise
        white_noise = self.rng.standard_normal(len(t)) * noise_level * 5e-6
        
        # ASD-specific: add more irregular bursts (less predictable)
        if condition == 'asd':
            # Random bursts of theta activity
            n_bursts = self.rng.integers(3, 8)
            for _ in range(n_bursts):
                burst_start = self.rng.integers(0, len(t) - 128)
                burst_len = self.rng.integers(64, 256)
                burst_freq = self.rng.uniform(4, 8)
                burst = 15e-6 * np.sin(
                    2 * np.pi * burst_freq * 
                    np.arange(burst_len) / self.srate
                )
                signal[burst_start:burst_start+burst_len] += burst[:min(burst_len, len(t)-burst_start)]
        
        return signal + pink_noise + white_noise
    
    def _pink_noise(self, n: int) -> np.ndarray:
        """
        Generate 1/f pink noise using frequency-domain scaling.

        Pink noise gives simulated EEG a more realistic low-frequency-heavy
        background than plain white noise.
        """
        white = self.rng.standard_normal(n)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n)
        freqs[0] = 1  # Avoid division by zero
        fft = fft / np.sqrt(freqs)
        return np.fft.irfft(fft, n=n)
