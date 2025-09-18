

import sys
import csv
import atexit
import asyncio
import numpy as np
import pandas as pd
from time import time, sleep
import signal
from multiprocessing import Process, Queue, Event
from datetime import datetime

from proxies.MuseProxy import MuseProxy
from utils.audio import soft_beep #not used for this experiement
#from utils.visualization import visualizer #not used for this experiement

from experiment.AudioBiofeedbackEngine import AudioBioFeedbackEngine

MUSE_MAC_ADDRESS = "00:55:DA:B6:21:8A" #replace with your own MAC address


"""
CSV file handling
"""

now = datetime.now()
timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
filename = f"{timestamp_str}_alphaWaveExp"
annotation_filename = f"{timestamp_str}_annotations"

eeg_file = open(f"data/{filename}.csv", "w", newline="")
eeg_writer = csv.writer(eeg_file)
eeg_writer.writerow(["Timestamp", "TP9", "AF7", "AF8", "TP10"])


annotation_file = open(f"data/{annotation_filename}.csv", "w", newline="")
annotation_writer = csv.writer(annotation_file)
annotation_writer.writerow(["Timestamp", "Cue"])

experienceEngine = None


def close_file():
    eeg_file.close()
    annotation_file.close()

atexit.register(close_file)


"""
EEG call back

"""
def eeg_callback(timestamps, data):
    """
    And to the file
    """
    global experienceEngine

    sampleTime = time()
    for i in range(data.shape[0]):
        eeg_writer.writerow([sampleTime-(11-i)*(1/256)] + data[i,:].tolist())


    if experienceEngine is None:
        experienceEngine = AudioBioFeedbackEngine()
        experienceEngine.start()
    else:
        experienceEngine.add_samples(data)
    
    
    
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

    ONLINE_RUN = True

    try:
        # Initialize Muse connection
        proxy = None

        if ONLINE_RUN:
            proxy = MuseProxy(MUSE_MAC_ADDRESS, eeg_callback)
            proxy.waitForConnected()
        else:

            eeg_data = pd.read_csv("data/2025-07-26_23-43-23_alphaWaveExp.csv")

            data = eeg_data.values
            data = data[:, 1:]

            batch_size = 10
            for i in range(0, data.shape[0], batch_size):
                samplesArray = data[i:i + batch_size]
                eeg_callback(None, samplesArray)

        # Optional buffer period to stabilize signal
        print("Initial padding, to stabilize signals...")
        sleep(20)

        # Start of live demo sequence
        print("Starting experience")
        sleep(360)
        print("Experience End")

        #experienceEngine.showResults()

    except KeyboardInterrupt:
        print("[MAIN] KeyboardInterrupt caught in try-block.")

    finally:
        # Ensure clean shutdown
        print("[MAIN] Cleaning up...")
        shutdown_event.set()        # Signal visualizer to stop
        if proxy is not None:
            proxy.disconnect()          # Disconnect Muse
        print("[MAIN] Shutdown complete. Goodbye!")