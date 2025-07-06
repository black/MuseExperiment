
import csv
import atexit
import asyncio
import numpy as np
from time import time, sleep
import signal
import sounddevice as sd

from MuseProxy import MuseProxy

MUSE_MAC_ADDRESS = "00:55:DA:B3:CC:35"  #Replace with your own mac address

# File setup
eeg_file = open("eeg_data.csv", "w", newline="")
eeg_writer = csv.writer(eeg_file)
eeg_writer.writerow(["Timestamp", "TP9", "AF7", "AF8", "TP10"])

def close_file():
    eeg_file.close()

atexit.register(close_file)

def eeg_callback(timestamps, data):
    for i in range(data.shape[0]):
        eeg_writer.writerow([timestamps[i]] + data[i, :].tolist())


def soft_beep(frequency=440, duration=0.5, volume=0.2, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = volume * np.sin(2 * np.pi * frequency * t)
    sd.play(wave, samplerate=sample_rate)
    sd.wait()

if __name__ == "__main__":
    
        
    connected = False
    
    proxy = MuseProxy(MUSE_MAC_ADDRESS, eeg_callback)
    proxy.waitForConnected()
    
    sleep(1)  
    print("Initial padding, to stabilize signals")
    sleep(20)   

    print("Starting experience")

    print("Phase 1 - eye blinks - blink on trigger (3 seconds, 5 times)")
    soft_beep()

    for i in range(5):
        sleep(3)
        soft_beep()
        print("blink")
        
    sleep(3)
    print("Done")
    sleep(2)

    soft_beep()
    print("Phase 2 - do nothing for 30 seconds (stay out of focus distracted)")
    sleep(30)

    soft_beep()
    print("Phase 3 - read for 30 seconds")
    sleep(30)

    print("Experiment over")

    proxy.disconnect()
     