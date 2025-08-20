# ==============================
# EEG Alpha Wave Experiment with Online Normal Estimator (z-score model)
# ==============================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Project-specific utilities
from utils.filters import bandpass_filter
from utils.eeg_utils import *
from utils.data_parser import *
from utils.models import *

# Visualization helpers
from utils.ep3_visualization import *


# --------------------------
# Demo Plots for Normal Distribution & z-Score
# --------------------------
plot_single_gaussian_demo()
demo_zScore()


# --------------------------
# Load and Preprocess Data
# --------------------------

# File paths for EEG and annotations
eeg_filename = 'data/2025-07-26_23-43-23_alphaWaveExp.csv'
annotation_filename = 'data/2025-07-26_23-43-23_annotations.csv'

# Load EEG and annotation data
eeg_data = pd.read_csv(eeg_filename)
annotations = pd.read_csv(annotation_filename)

# Extract and normalize timestamps
timestamps = eeg_data['Timestamp'].values
start = timestamps[0]
timestamps = np.arange(0, len(timestamps)) * (1 / 256) + start  # uniform timeline (fs = 256 Hz)

# Select EEG channels of interest
channels = ['TP9', 'AF7', 'AF8', 'TP10']
eeg_signals = eeg_data[channels].values

# Constants
fs = 256                # Sampling frequency (Hz)
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

# Extract trial windows from annotations
eye_open_trials = extract_trials(annotations, 'eye-open-begin', TRIAL_LENGTH_SEC)
eye_closed_trials = extract_trials(annotations, 'eye-closed-begin', TRIAL_LENGTH_SEC)

# Treat entire recording as one long "trial" for continuous analysis
trials = [[0, timestamps.max()]]

# Compute FFT-based power spectra for all segments (with overlap)
segments = compute_all_fft_segments(
    trials, timestamps, filtered_data,
    trial_length=int(timestamps.max()), overlap=0.25
)

# Compute relative alpha power for each segment
relAlpha = compute_relative_alpha(segments)


# --------------------------
# Online Normal Estimator (z-score model)
# --------------------------

from matplotlib import gridspec

THRESHOLD_OUTPUT = True  # If True, output binary labels; otherwise continuous z-scores

# Initialize online estimator
zscoreModel = OnlineNormalEstimator(mu_prior=0.2, sigma_prior=0.2, eta=0.005)

# Track model parameters
means, sigmas, label = [], [], []

for value in relAlpha:
    # Update model with new observation
    zscoreModel.update(value)
    params = zscoreModel.get_model_parameters()
    
    means.append(params["mu"])
    sigmas.append(params["sigma"])   # Correct: store sigma values
    
    # Threshold or raw output
    if THRESHOLD_OUTPUT:
        if zscoreModel.predict(value) < 0:   # anomaly detected
            label.append(-0.25)
        else:
            label.append(0)
    else:
        label.append(zscoreModel.predict(value))

# Convert to numpy arrays for plotting
means = np.array(means)
sigmas = np.array(sigmas)
label = np.array(label)


# --------------------------
# Visualization
# --------------------------

sns.set_style("whitegrid")
PALETTE = sns.color_palette("husl", 2)

# Final estimated parameters
final_mu = means[-1]
final_sigma = sigmas[-1]

# Grid for Gaussian distribution (not directly plotted here)
x_grid = np.linspace(final_mu - 3 * final_sigma,
                     final_mu + 3 * final_sigma, 500)

# --- Set up figure with gridspec ---
fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)

# --- Main time-series axis ---
ax0 = plt.subplot(gs[0])
nbSamples = means.shape[0]
time_axis = np.linspace(0, 1, nbSamples) * 1227  # scaled time index

# Plot normal distribution mean ± sigma over time
ax0.plot(time_axis, means, color='black', label="Normal Dist. Mean")
ax0.fill_between(time_axis,
                 means - sigmas,
                 means + sigmas,
                 color='black', alpha=0.2)

# Plot labels (thresholded or continuous)
ax0.plot(time_axis, label, color='k', label="Label")

# Smooth label with moving average
window = 15
kernel = np.ones(window) / window
smoothed_label = np.convolve(label, kernel, mode="same")
ax0.plot(time_axis, smoothed_label, color='r', label="Smoothed Label")

# Shade eye-open trial intervals
for start, end in eye_open_trials:
    ax0.axvspan(start - timestamps[0], end - timestamps[0],
                color="lightblue", alpha=0.2, label="Eye open")

# Shade eye-closed trial intervals
for start, end in eye_closed_trials:
    ax0.axvspan(start - timestamps[0], end - timestamps[0],
                color="lightcoral", alpha=0.2, label="Eye closed")

# Labels and title
ax0.set_xlabel('Iteration')
ax0.set_ylabel('Normal means ± sigma')
ax0.set_title('Online Normal Distribution Tracking')

# Remove duplicate legend entries
handles, labels = ax0.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax0.legend(unique.values(), unique.keys(), loc='lower right')

plt.show()
