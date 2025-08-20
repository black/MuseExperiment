
import numpy as np
from utils.data_parser import *


def compute_avg_fft(
    trials, 
    timestamps, 
    filtered_data, 
    fs=256, 
    window_sec=2, 
    trial_length=120,
    channels=(0, 3)  # default channels
):
    """
    Compute average FFT over trials and windows, averaging power over selected channels.
    
    Parameters:
        trials: list of (start_time, end_time) tuples
        timestamps: array of timestamps
        filtered_data: ndarray of shape (samples, channels)
        fs: sampling frequency
        window_sec: FFT window length in seconds
        trial_length_samples: expected number of samples per trial
        channels: tuple/list of channel indices to average
        
    Returns:
        freqs: array of frequency bins
        avg_power: average power spectrum
    """
    trial_length_samples = trial_length*fs
    window_samples = int(window_sec * fs)
    fft_accum = []
    
    for start_time, end_time in trials:
        trial_data = get_trial_data(
            timestamps, filtered_data, start_time, end_time, trial_length_samples
        )
        
        # Use only selected channels
        trial_data = trial_data[:, channels]
        n_samples = trial_data.shape[0]
        n_windows = n_samples // window_samples
        
        for i in range(n_windows):
            segment = trial_data[i * window_samples:(i + 1) * window_samples, :]
            if segment.shape[0] < window_samples:
                continue
            
            # Window function applied to each channel
            windowed = segment * np.hanning(window_samples)[:, None]
            
            # FFT for each channel
            fft_vals = np.fft.rfft(windowed, axis=0)
            power = np.abs(fft_vals) ** 2  # shape: (freq_bins, n_channels)
            
            # Average power across channels
            mean_power_channels = np.mean(power, axis=1)
            fft_accum.append(mean_power_channels)
    
    freqs = np.fft.rfftfreq(window_samples, 1 / fs)
    avg_power = np.mean(fft_accum, axis=0) if fft_accum else np.zeros(len(freqs))
    
    return freqs, avg_power

def compute_all_fft_segments(
    trials, 
    timestamps, 
    filtered_data, 
    fs=256, 
    window_sec=2, 
    trial_length=120,
    channels=(0, 3),   # default channels to average
    overlap=0.0        # fraction of overlap between windows (0 = no overlap, 0.5 = 50%)
):
    """
    Compute FFT power for each segment in all trials, averaging across selected channels.
    
    Parameters:
        trials: list of (start_time, end_time) tuples
        timestamps: array of timestamps
        filtered_data: ndarray of shape (samples, channels)
        fs: sampling frequency
        window_sec: FFT window length in seconds
        trial_length: expected trial length in seconds
        channels: tuple/list of channel indices to average
        overlap: fraction [0,1) controlling overlap between windows.
                 0 = no overlap, 0.5 = 50% overlap, etc.
        
    Returns:
        all_powers: ndarray of shape (n_segments, n_freq_bins)
    """
    trial_length_samples = trial_length * fs
    window_samples = int(window_sec * fs)
    step_size = int(window_samples * (1 - overlap))  # shift between windows
    if step_size <= 0:
        raise ValueError("overlap too high; must be <1.0")

    all_powers = []
    
    for start_time, end_time in trials:
        trial_data = get_trial_data(
            timestamps, filtered_data, start_time, end_time, trial_length_samples
        )
        
        # Select only the desired channels
        trial_data = trial_data[:, channels]
        n_samples = trial_data.shape[0]
        
        # Sliding windows
        for start in range(0, n_samples - window_samples + 1, step_size):
            segment = trial_data[start:start + window_samples, :]
            if segment.shape[0] < window_samples:
                continue
            
            # Apply window function to each channel
            windowed = segment * np.hanning(window_samples)[:, None]
            
            # FFT for each channel
            fft_vals = np.fft.rfft(windowed, axis=0)
            power = np.abs(fft_vals) ** 2  # shape: (freq_bins, n_channels)
            
            # Average power across channels
            mean_power_channels = np.mean(power, axis=1)
            all_powers.append(mean_power_channels)
    
    return np.array(all_powers)



def compute_relative_alpha(powers, fs=256, window_sec=2, alpha_band=(8, 13)):
    freqs = np.fft.rfftfreq(int(fs * window_sec), d=1/fs)
    alpha_idx = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
    relative_alpha = []
    for p in powers:
        total_power = np.sum(p)
        alpha_power = np.sum(p[alpha_idx])
        relative_alpha.append(alpha_power / total_power)
    return np.array(relative_alpha)