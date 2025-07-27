"""
    Episode 2 - Alpha Wave experiement

    Author: Fred Simard, RE-AK Technologies, 2025
    Website: https://www.re-ak.com
    Discord: https://discord.gg/XzeDHJf6ne

    Description:
    ------------
    This script is part of a tutorial series teaching the basics of hacking with brain-computer interfaces.
    
    It drives an alpha wave experiment

    This is a foundational script intended for educational use and is available on GitHub as part of a
    YouTube tutorial series.

    Core Features:
    --------------
    - Data and annotation collection
    - Experiment design and cued user instructions
    - Basic feature extraction and statistical analysis

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
from utils.audio import soft_beep #not used for this experiement
#from utils.visualization import visualizer #not used for this experiement

MUSE_MAC_ADDRESS = "00:55:DA:B3:CC:35" #replace with your own MAC address


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


def close_file():
    eeg_file.close()
    annotation_file.close()

atexit.register(close_file)


"""
EEG call back

Sends data to QT vizualizer and to file

"""
def eeg_callback(timestamps, data):
    """
    And to the file
    """
    sampleTime = time()
    for i in range(data.shape[0]):
        eeg_writer.writerow([sampleTime-(11-i)*(1/256)] + data[i,:].tolist())
    
    
    
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

    try:
        # Initialize Muse connection
        proxy = MuseProxy(MUSE_MAC_ADDRESS, eeg_callback)
        proxy.waitForConnected()

        # Optional buffer period to stabilize signal
        print("Initial padding, to stabilize signals...")
        sleep(20)

        # Start of live demo sequence
        print("Starting experience")

        # Phase 1: Just let the EEG stream for 6 minutes (adjust to your demo needs)
        for i in range(4):
        
            print("")
            print(f"-------------------")
            print(f"     Trial {i}     ")
            print(f"-------------------")
            
            print(f"30 seconds - Pre-trial period, blink a few times")
            soft_beep()
            sleep(30)
            
            annotation_writer.writerow([time(), "eye-open-begin"])
            print(f"120 seconds - eye-open - reading")
            sleep(120)
            
            annotation_writer.writerow([time(), "eye-open-end"])
            soft_beep()
            print(f"30 seconds - inter-condition padding, close your eyes")
            sleep(30)
            
            annotation_writer.writerow([time(), "eye-closed-begin"])
            print(f"120 seconds - eye-closed - relaxation")
            sleep(120)
            
            annotation_writer.writerow([time(), "eye-closed-end"])
            
        

    except KeyboardInterrupt:
        print("[MAIN] KeyboardInterrupt caught in try-block.")

    finally:
        # Ensure clean shutdown
        print("[MAIN] Cleaning up...")
        shutdown_event.set()        # Signal visualizer to stop
        proxy.disconnect()          # Disconnect Muse
        print("[MAIN] Shutdown complete. Goodbye!")