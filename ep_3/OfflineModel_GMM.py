# ==============================
# EEG Alpha Wave Experiment with Online GMM Tracking
# ==============================

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Signal processing & statistics
from scipy.signal import butter, filtfilt

# Evaluation metrics
from sklearn.metrics import roc_curve, roc_auc_score

# Project-specific utilities
from utils.filters import bandpass_filter
from utils.eeg_utils import *
from utils.data_parser import *
from utils.models import *

# Visualization utilities
from utils.ep3_visualization import *


# --------------------------
# Demo Plots for Likelihood & GMM
# --------------------------
plot_two_class_likelihood_demo()
demo_GMM()


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
# Generate uniform timeline (fs = 256 Hz)
timestamps = np.arange(0, len(timestamps)) * (1 / 256) + start

# Select EEG channels of interest
channels = ['TP9', 'AF7', 'AF8', 'TP10']
eeg_signals = eeg_data[channels].values

# Constants
fs = 256  # Sampling frequency (Hz)
TRIAL_LENGTH_SEC = 120  # Trial duration

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

# For continuous analysis, take whole recording as one "trial"
trials = [[0, timestamps.max()]]

# Compute FFT power spectra across segments (with overlap)
segments = compute_all_fft_segments(
    trials, timestamps, filtered_data,
    trial_length=int(timestamps.max()), overlap=0.25
)

# Compute relative alpha power for each segment
relAlpha = compute_relative_alpha(segments)


# --------------------------
# Online Two-State GMM Fitting
# --------------------------

from matplotlib import gridspec
from scipy.stats import norm

# Initialize Online Two-State GMM
gmm = OnlineTwoStateGMM(
    mu_prior=[0.05, 0.3],      # Initial means
    sigma_prior=[0.2, 0.2],    # Initial standard deviations
    pi_prior=[0.5, 0.5],       # Initial mixture weights
    eta=0.01,                  # Learning rate
    pi_min=0.05                # Minimum mixture weight
)

# Track model evolution
means, sigmas, label = [], [], []

for value in relAlpha:
    gmm.update(value)
    params = gmm.get_model_parameters()
    
    means.append(params["mu"])
    sigmas.append(params["sigma"])   # <-- Correct: store sigma, not mu
    
    # Predicted label: compare probabilities of the two components
    if gmm.predict(value)[0] > gmm.predict(value)[1]:
        label.append(-0.25)   # Assign small negative offset for plotting
    else:
        label.append(0)

# Convert to arrays for plotting
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

# Grid for Gaussian distributions (not directly plotted here)
x_grid = np.linspace(
    min(final_mu) - 3 * max(final_sigma),
    max(final_mu) + 3 * max(final_sigma), 500
)

# --- Set up figure with two-panel layout ---
fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)

# --- Main time-series axis ---
ax0 = plt.subplot(gs[0])
nbSamples = means.shape[0]
time_axis = np.linspace(0, 1, nbSamples) * 1227  # Scaled time index

# Plot component 1 mean ± sigma
ax0.plot(time_axis, means[:, 0], color=PALETTE[0], label="Component 1")
ax0.fill_between(time_axis,
                 means[:, 0] - sigmas[:, 0],
                 means[:, 0] + sigmas[:, 0],
                 color=PALETTE[0], alpha=0.2)

# Plot component 2 mean ± sigma
ax0.plot(time_axis, means[:, 1], color=PALETTE[1], label="Component 2")
ax0.fill_between(time_axis,
                 means[:, 1] - sigmas[:, 1],
                 means[:, 1] + sigmas[:, 1],
                 color=PALETTE[1], alpha=0.2)

# Plot raw label predictions
ax0.plot(time_axis, label, color='k', label="Label")

# Smooth label curve with moving average
window = 15   # smoothing window size
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
ax0.set_ylabel('GMM means ± sigma')
ax0.set_title('Online GMM Tracking')

# Remove duplicate legend entries
handles, labels = ax0.get_legend_handles_labels()
unique = dict(zip(labels, handles))
# ax0.legend(unique.values(), unique.keys(), loc='lower right')

plt.show()
