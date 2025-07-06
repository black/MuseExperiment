# MuseExperiment

This repository provides a simple experimental setup for the **Muse** EEG headband by Interaxon, as demonstrated in the accompanying Youtube video (not online, yet)

## 🧠 Overview

This experiment connects to a Muse device using a custom `MuseProxy` object, which handles the software interface and includes an **auto-reconnect** feature for reliability.

## 🗓️ Status – July 5, 2025

Currently, the repository includes two files:

- `MuseProxy.py` – Implements the connection and data handling logic for the Muse device.
- `experimentManager.py` – Launches the experiment session and manages data collection.

### 🚀 Running the Experiment

To get started, simply:

- edit `experimentManager.py` to set your Muse Mac address

```bash
cd application
python experimentManager.py
```

This will initiate a short experience session. All EEG data will be saved to
eeg_data.csv

### Running on Linux
It should work, but you need to remove the soft_beep method as it's specific for Windows.
