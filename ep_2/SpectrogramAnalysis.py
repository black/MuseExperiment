import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.filters import bandpass_filter
from scipy.signal import butter, filtfilt, spectrogram

# --------------------------
# Load and Preprocess Data
# --------------------------

# File paths
eeg_filename = 'data/2025-07-26_23-43-23_alphaWaveExp.csv'
annotation_filename = 'data/2025-07-26_23-43-23_annotations.csv'

# Load EEG and annotation data
eeg_data = pd.read_csv(eeg_filename)
annotations = pd.read_csv(annotation_filename)

# Sampling frequency (Hz)
fs = 256

# Time vector: evenly spaced assuming 256 Hz sampling
timestamps = eeg_data['Timestamp'].values
start_time = timestamps[0]
timestamps = np.arange(0, len(timestamps)) * (1 / fs) + start_time

# EEG channel selection
channels = ['TP9', 'AF7', 'AF8', 'TP10']
eeg_signals = eeg_data[channels].values

# Remove DC offset from each channel
eeg_signals -= eeg_signals.mean(axis=0)

# Apply bandpass filter (0.5–40 Hz) to each channel
lowcut, highcut = 0.5, 40.0
filtered_data = np.array([
    bandpass_filter(eeg_signals[:, i], lowcut, highcut, fs)
    for i in range(eeg_signals.shape[1])
]).T

# --------------------------
# Plot Spectrograms
# --------------------------

plt.figure(figsize=(10, 8))

# Loop through each channel and generate a spectrogram
for i, channel in enumerate(channels):
    plt.subplot(len(channels), 1, i + 1)
    
    # Compute the spectrogram with a 1-second window (nperseg=fs)
    f, t, Sxx = spectrogram(filtered_data[:, i], fs, nperseg=fs)
    
    # Convert power to dB scale
    Sxx_db = 10 * np.log10(Sxx)
    
    # Z-score normalization across time (for each frequency bin)
    mean_Sxx = np.mean(Sxx_db, axis=1)
    std_Sxx = np.std(Sxx_db, axis=1)
    z_score_Sxx = (Sxx_db - mean_Sxx[:, None]) / std_Sxx[:, None]
    
    # Plot normalized spectrogram
    plt.pcolormesh(t, f, z_score_Sxx, shading='auto', cmap='gist_heat')
    plt.ylabel('Frequency [Hz]')
    plt.title(f'Spectrogram of {channel} (Z-Score Normalized dB)')
    plt.ylim(0, 40)

    # Add vertical lines at annotation timestamps
    for _, row in annotations.iterrows():
        time_offset = row['Timestamp'] - start_time
        if row['Cue'] in ['eye-open-begin', 'eye-open-end']:
            plt.axvline(x=time_offset, color='white', linewidth=2)
        elif row['Cue'] in ['eye-closed-begin', 'eye-closed-end']:
            plt.axvline(x=time_offset, color='blue', linewidth=2)

# Add common x-axis label
plt.xlabel('Time [s]')
plt.tight_layout()
plt.show()
