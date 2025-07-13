"""
    EEG Real-Time Recording and Visualization with Muse (Interaxon)

    Author: Fred Simard, RE-AK Technologies, 2025
    Website: https://www.re-ak.com
    Discord: https://discord.gg/XzeDHJf6ne

    Description:
    ------------
    This script is part of a tutorial series teaching how to use the Muse EEG headset by Interaxon.
    It demonstrates how to acquire real-time EEG data, save raw signals to a CSV file, and visualize
    filtered EEG in a live plot. The live demo shows common signal artifacts including eye blinks,
    jaw clenches, frowning, and line noise.

    This is a foundational script intended for educational use and is available on GitHub as part of a
    YouTube tutorial series.

    Core Features:
    --------------
    - Real-time EEG data acquisition using MuseProxy
    - Raw data saved to CSV
    - Real-time bandpass-filtered visualization (0.5–40 Hz)
    - Soft audio beeps to guide user actions
    - Graceful handling of Ctrl+C to ensure proper shutdown

    Dependencies:
    -------------
    - Tested on Python 3.12
    - pyqtgraph
    - numpy
    - scipy
    - sounddevice (unused)
    - MuseProxy module (custom)
"""


import sys
import csv
import atexit
import asyncio
import numpy as np
from time import time, sleep
import signal
from multiprocessing import Process, Queue, Event
from datetime import datetime

from proxies.MuseProxy import MuseProxy
#from utils.audio import soft_beep #not used for this experiement
from utils.visualization import visualizer

MUSE_MAC_ADDRESS = "00:55:DA:B3:CC:35" #replace with your own MAC address


"""
CSV file handling
"""

now = datetime.now()
timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
filename = f"my_file_{timestamp_str}.txt"

eeg_file = open(f"data/{filename}.csv", "w", newline="")
eeg_writer = csv.writer(eeg_file)
eeg_writer.writerow(["Timestamp", "TP9", "AF7", "AF8", "TP10"])

def close_file():
    eeg_file.close()

atexit.register(close_file)

"""
Ctrl-C signal handling
"""
shutdown_event = Event()

def signal_handler(sig, frame):
    print("\n[INFO] Caught Ctrl+C, shutting down gracefully...")
    shutdown_event.set()

signal.signal(signal.SIGINT, signal_handler)

"""
EEG call back

Sends data to QT vizualizer and to file

"""
def eeg_callback(timestamps, data):
    """
    Called when new EEG samples are received from the Muse.
    Pushes the data to the visualization queue for plotting.
    """
    q.put(data)  # send to visualizer
    
    """
    And to the file
    """
    for i in range(data.shape[0]):
        eeg_writer.writerow([timestamps[i]] + data[i,:].tolist())
    
    
    
# ------------------------------
# Graceful Shutdown Handling
# ------------------------------

# Event object to coordinate shutdown across processes
shutdown_event = Event()

def signal_handler(sig, frame):
    """
    Handles Ctrl+C (SIGINT) to trigger a clean shutdown.
    """
    print("\n[INFO] Ctrl+C received. Initiating shutdown...")
    shutdown_event.set()

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

    
# ------------------------------
# Main Execution
# ------------------------------

if __name__ == "__main__":
    # Queue for inter-process communication (EEG data)
    q = Queue()

    # Start visualization in a separate process
    vis_process = Process(target=visualizer, args=(q, shutdown_event))
    vis_process.start()

    try:
        # Initialize Muse connection
        proxy = MuseProxy(MUSE_MAC_ADDRESS, eeg_callback)
        proxy.waitForConnected()

        # Optional buffer period to stabilize signal
        sleep(1)
        print("Initial padding, to stabilize signals...")
        sleep(20)

        # Start of live demo sequence
        print("Starting experience")

        # Phase 1: Just let the EEG stream for 6 minutes (adjust to your demo needs)
        sleep(360)

    except KeyboardInterrupt:
        print("[MAIN] KeyboardInterrupt caught in try-block.")

    finally:
        # Ensure clean shutdown
        print("[MAIN] Cleaning up...")
        shutdown_event.set()        # Signal visualizer to stop
        proxy.disconnect()          # Disconnect Muse
        q.put(None)
        vis_process.terminate()     # Force-stop the visualizer (in case it lags)
        vis_process.join()          # Wait for visualizer to exit
        print("[MAIN] Shutdown complete. Goodbye!")