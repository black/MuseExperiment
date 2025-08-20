# ==============================
# EEG Alpha Wave Experiment Analysis
# ==============================

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Signal processing & statistics
from scipy.signal import butter, filtfilt

# Evaluation metrics
from sklearn.metrics import roc_curve, roc_auc_score

# Project-specific utilities
from utils.filters import bandpass_filter
from utils.eeg_utils import *
from utils.data_parser import *

# Visualization functions
from utils.ep3_visualization import *


# --------------------------
# Load and Preprocess Data
# --------------------------

# File paths for EEG signals and annotation markers
eeg_filename = 'data/2025-07-26_23-43-23_alphaWaveExp.csv'
annotation_filename = 'data/2025-07-26_23-43-23_annotations.csv'

# Load EEG and annotation data
eeg_data = pd.read_csv(eeg_filename)
annotations = pd.read_csv(annotation_filename)

# Extract and normalize timestamps
timestamps = eeg_data['Timestamp'].values
start = timestamps[0]
# Create uniform timeline assuming fs = 256 Hz
timestamps = np.arange(0, len(timestamps)) * (1 / 256) + start

# Select EEG channels of interest
channels = ['TP9', 'AF7', 'AF8', 'TP10']
eeg_signals = eeg_data[channels].values

# Constants
fs = 256  # Sampling frequency (Hz)
TRIAL_LENGTH_SEC = 120  # Trial duration in seconds

# Apply bandpass filter (0.5–40 Hz) to each EEG channel
lowcut, highcut = 0.5, 40.0
filtered_data = np.array([
    bandpass_filter(eeg_signals[:, i], lowcut, highcut, fs)
    for i in range(eeg_signals.shape[1])
]).T


# --------------------------
# Segment Data into Conditions
# --------------------------

# Extract trial windows based on annotation events
eye_open_trials = extract_trials(annotations, 'eye-open-begin', TRIAL_LENGTH_SEC)
eye_closed_trials = extract_trials(annotations, 'eye-closed-begin', TRIAL_LENGTH_SEC)

# Compute average FFT power spectra for both conditions
freqs_open, power_open = compute_avg_fft(eye_open_trials, timestamps, filtered_data)
freqs_closed, power_closed = compute_avg_fft(eye_closed_trials, timestamps, filtered_data)


# --------------------------
# Smooth and Plot Power Spectra
# --------------------------

# Currently no Gaussian smoothing applied (kept raw power)
power_open_smooth = power_open
power_closed_smooth = power_closed

# Plot smoothed (or raw) power spectra for both conditions
plot_power_spectra(freqs_open, power_open, freqs_closed, power_closed)

# Compute FFT power spectra for all segments (not averaged)
powers_open = compute_all_fft_segments(eye_open_trials, timestamps, filtered_data)
powers_closed = compute_all_fft_segments(eye_closed_trials, timestamps, filtered_data)


# --------------------------
# Alpha Band Analysis
# --------------------------

# Define frequency axis and alpha band (8–13 Hz)
freqs = np.fft.rfftfreq(int(fs * 2), d=1/fs)
alpha_band = (8, 13)

# Compute relative alpha power for both conditions
rel_alpha_open = compute_relative_alpha(powers_open)
rel_alpha_closed = compute_relative_alpha(powers_closed)

# Build label vector: 0 = eye open, 1 = eye closed
y_true = np.concatenate([
    np.zeros(len(rel_alpha_open)),
    np.ones(len(rel_alpha_closed))
])

# Scores are the relative alpha values
scores = np.concatenate([rel_alpha_open, rel_alpha_closed])

# ROC curve and AUC evaluation
fpr, tpr, thresholds = roc_curve(y_true, scores)
auc_value = roc_auc_score(y_true, scores)

# Compute condition-wise means
mean_open = np.mean(rel_alpha_open)
mean_closed = np.mean(rel_alpha_closed)

# Plot histograms of alpha power distributions + ROC curve
plot_alpha_distribution(
    rel_alpha_open, rel_alpha_closed,
    mean_open, mean_closed,
    auc_value, fpr, tpr
)

# Plot an explanation figure showing variability
plot_std_explanation(rel_alpha_open)
