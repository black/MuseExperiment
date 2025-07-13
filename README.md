# 🧠 MuseExperiment – EEG Visual Demos with Muse by Interaxon

**Author:** Fred Simard, [RE-AK Technologies](https://www.re-ak.com), 2025  
📣 [Join our Discord](https://discord.gg/XzeDHJf6ne) for support, discussion, and updates!

---

This repository provides a minimal experimental setup to get started with the **Muse** EEG headband by **Interaxon**. It's designed to accompany a YouTube tutorial series (Lesson 1 coming soon), focusing on real-time EEG signal acquisition and artifact visualization.

---

## 📦 Contents

- ✅ **`ep_1/`** – A beginner-friendly example that:
  - Connects to a Muse headset
  - Starts a real-time visualizer using PyQtGraph
  - Saves filtered EEG data to CSV
  - Demonstrates common EEG artifacts: blinks, jaw clenches, frowning, noise

- ✅ **`MuseProxy`** – A robust, auto-reconnecting BLE handler that simplifies Muse communication using `bleak` and `asyncio`.

---

## 🚀 Getting Started

### Requirements

Make sure you have the following installed:

- edit `experimentManager.py` to set your Muse Mac address

```bash
pip install requirements.txt
```

## 🔍 Find Your Muse MAC Address

```bash
cd ep_1
python ble_scanner.py
```

## ▶️ Run the Live Visualizer

```bash
cd ep_1
python lesson_1.py
```

This will:

Connect to your Muse headset

Display real-time EEG plots

Save data to:
ep_1/data/my_file_{timestamp}.csv

### Running on Linux
It should work, but you need to remove the soft_beep method as it's specific for Windows.


## 🛠️ Status – July 13, 2025
This is Lesson 1 in a planned multi-part series. It is functional and tested with the Muse 2 and Muse S devices. Contributions and issues are welcome.


