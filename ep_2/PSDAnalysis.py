import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from utils.filters import bandpass_filter
import seaborn as sns

# Extract trials based on cue label
def extract_trials(annotations, cue_start):
    """
    Extract start and end times for trials based on cue type.
    
    Parameters:
        annotations (DataFrame): Annotation data with 'Cue' and 'Timestamp' columns.
        cue_start (str): Cue string to identify trial start.

    Returns:
        list of tuples: Each tuple contains (start_time, end_time) for a trial.
    """
    trials = []
    for _, row in annotations.iterrows():
        if cue_start in row['Cue']:
            start_time = row['Timestamp']
            end_time = start_time + TRIAL_LENGTH_SEC
            trials.append((start_time, end_time))
    return trials

# Slice trial data from filtered EEG using timestamps
def get_trial_data(start_time, end_time):
    """
    Get filtered EEG data for a specific trial window.
    
    Parameters:
        start_time (float): Trial start timestamp.
        end_time (float): Trial end timestamp.

    Returns:
        ndarray: EEG data for the specified trial.
    """
    start_idx = np.searchsorted(timestamps, start_time)
    end_idx = start_idx + TRIAL_LENGTH_SAMPLES
    return filtered_data[start_idx:end_idx, :]

# Compute average power spectrum across trials using FFT
def compute_avg_fft(trials):
    """
    Compute average FFT power spectrum for a list of trials.
    
    Parameters:
        trials (list of tuples): Each tuple is (start_time, end_time).

    Returns:
        tuple: Frequencies and corresponding average power spectrum.
    """
    fft_accum = []
    for start_time, end_time in trials:
        trial_data = get_trial_data(start_time, end_time)
        signal = trial_data[:, 0]  # Analyze first channel
        n = len(signal)
        freqs = np.fft.rfftfreq(n, 1/fs)
        fft_vals = np.fft.rfft(signal * np.hanning(n))  # Windowed FFT
        power = np.abs(fft_vals) ** 2
        fft_accum.append(power)
    avg_power = np.mean(fft_accum, axis=0)
    return freqs, avg_power

# --------------------------
# Load and Preprocess Data
# --------------------------

# File paths for EEG and annotations
eeg_filename = 'data/2025-07-26_23-43-23_alphaWaveExp.csv'
annotation_filename = 'data/2025-07-26_23-43-23_annotations.csv'

# Load data
eeg_data = pd.read_csv(eeg_filename)
annotations = pd.read_csv(annotation_filename)

# Extract and normalize timestamps
timestamps = eeg_data['Timestamp'].values
start = timestamps[0]
timestamps = np.arange(0, len(timestamps)) * (1 / 256) + start

# Select EEG channels and center data
channels = ['TP9', 'AF7', 'AF8', 'TP10']
eeg_signals = eeg_data[channels].values
eeg_signals -= eeg_signals.mean(axis=0)  # Remove DC offset per channel

fs = 256  # Sampling frequency (Hz)
TRIAL_LENGTH_SEC = 120
TRIAL_LENGTH_SAMPLES = int(TRIAL_LENGTH_SEC * fs)

# Apply bandpass filter to each EEG channel
lowcut, highcut = 0.5, 40.0
filtered_data = np.array([
    bandpass_filter(eeg_signals[:, i], lowcut, highcut, fs)
    for i in range(eeg_signals.shape[1])
]).T

# --------------------------
# Segment Data into Conditions
# --------------------------

# Extract trial windows based on annotations
eye_open_trials = extract_trials(annotations, 'eye-open-begin')
eye_closed_trials = extract_trials(annotations, 'eye-closed-begin')

# Compute FFT for both conditions
freqs_open, power_open = compute_avg_fft(eye_open_trials)
freqs_closed, power_closed = compute_avg_fft(eye_closed_trials)

# --------------------------
# Smooth and Plot Power Spectra
# --------------------------

# Apply Gaussian smoothing
power_open_smooth = power_open
power_closed_smooth = power_closed

for i in range(3):
    power_open_smooth = gaussian_filter1d(power_open_smooth, sigma=2)
    power_closed_smooth = gaussian_filter1d(power_closed_smooth, sigma=2)


# Set Seaborn style and color palette
sns.set_style("whitegrid")
palette = sns.color_palette("husl", 2)

# Plot average power spectra
plt.figure(figsize=(12, 7))

# Highlight the alpha band (8–13 Hz)
plt.axvspan(8, 13, color='gray', alpha=0.2, label='Alpha Band (8–13 Hz)')

plt.plot(freqs_open, power_open_smooth, label='Eye Open', color=palette[0], linewidth=1.5)
plt.plot(freqs_closed, power_closed_smooth, label='Eye Closed', color=palette[1], linewidth=1.5)

plt.xlabel('Frequency (Hz)', fontsize=14)
plt.ylabel('Power (amplitude)', fontsize=14)
plt.title('Average Power Spectrum by Condition (First Channel)', fontsize=16, weight='bold')

plt.xlim(0, 40)
plt.ylim(bottom=min(min(power_open_smooth), min(power_closed_smooth)) - 5)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper right', frameon=True, fancybox=True, shadow=True)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

sns.despine()
plt.tight_layout()
plt.show()
