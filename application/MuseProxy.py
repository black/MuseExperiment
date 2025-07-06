
import threading
import csv
import atexit
import asyncio
import numpy as np
import bitstring
from bleak import BleakClient, BleakError
from time import time, sleep
import platform
import subprocess
import signal

MUSE_MAC_ADDRESS = "00:55:DA:B3:CC:35"  #Replace with your own mac address

MUSE_GATT_ATTRS = {
    "STREAM_TOGGLE": '273e0001-4c4d-454d-96be-f03bac821358',
    "TP9": '273e0003-4c4d-454d-96be-f03bac821358',
    "AF7": '273e0004-4c4d-454d-96be-f03bac821358',
    "AF8": '273e0005-4c4d-454d-96be-f03bac821358',
    "TP10": '273e0006-4c4d-454d-96be-f03bac821358',
}

SAMPLE_RATE = 256


class MuseProxy:
    
    def __init__(self, mac_address, eeg_callback):
        self.muse = MuseBleak(mac_address, eeg_callback)
        self.runner = AsyncRunnerThread(main, self.muse)

        def handle_exit(signum=None, frame=None):
            print(f"Exiting... signal={signum}")
            self.runner.stop()

        # Call on Ctrl+C or kill
        signal.signal(signal.SIGINT, handle_exit)
        signal.signal(signal.SIGTERM, handle_exit)

        # Call on normal interpreter exit
        atexit.register(handle_exit)

        self.runner.start()
        
    def waitForConnected(self):
        connected = False
        while not connected:
            if self.muse.client:
                if self.muse.client.is_connected:
                    print(f"Connected")
                    connected = True
                else:
                    print(f"Waiting for Muse")
            else:
                print("Muse client not yet initialized.")
                
            sleep(1)
        
    def disconnect(self):
        self.muse.stop()
        
        

class MuseBleak:
    def __init__(self, address, callback_eeg, time_func=time):
        self.address = address
        self.callback_eeg = callback_eeg
        self.time_func = time_func
        self.should_run = True
        self.client = None
        self.sample_index = 0
        self.last_tm = None
        self.timestamps = np.zeros(4)
        self.data = np.zeros((4, 12))
        self.reg_params = np.array([self.time_func(), 1. / SAMPLE_RATE])

    async def connect_loop(self):
        """Loop to keep trying to connect and resume streaming."""
        while self.should_run:
            try:
                await self.connect()
                print("Connected successfully.")
                while self.client.is_connected and self.should_run:
                    await asyncio.sleep(1)
                print("Disconnected. Reconnecting...")
                self.last_tm = None
                
            except Exception as e:
                print(f"Connection error: {e}")
            await asyncio.sleep(3)  # Wait before trying to reconnect


    async def connect(self):
        self.client = BleakClient(self.address)
        await self.client.connect()

        for i, name in enumerate(["TP9", "AF7", "AF8", "TP10"]):
            await self.client.start_notify(
                MUSE_GATT_ATTRS[name],
                lambda sender, data, i=i: asyncio.create_task(self.handle_eeg(i, data))
            )

        await self.resume()

    async def resume(self):
        await self.client.write_gatt_char(MUSE_GATT_ATTRS["STREAM_TOGGLE"], bytearray([0x02, 0x64, 0x0a]))

    async def stop(self):
        self.should_run = False
        if self.client:
            if self.client.is_connected:
                try:
                    await self.client.write_gatt_char(
                        MUSE_GATT_ATTRS["STREAM_TOGGLE"],
                        bytearray([0x02, 0x68, 0x0a])
                    )
                    await self.client.disconnect()
                    print("Requested client disconnect.")
                    # Wait briefly to allow disconnect to propagate
                    await asyncio.sleep(1)
                except Exception as e:
                    print(f"BLE disconnect failed: {e}")
            
            # Double-check if still connected (common on Linux)
            if self.client.is_connected:
                print("Client still connected after disconnect attempt.")
                force_disconnect(self.address)
            else:
                print("Client disconnected cleanly.")

    def unpack_eeg(self, packet):
        aa = bitstring.Bits(bytes=packet)
        pattern = "uint:16," + ",".join(["uint:12"] * 12)
        res = aa.unpack(pattern)
        packet_index, data = res[0], res[1:]
        data = 0.48828125 * (np.array(data) - 2048)
        return packet_index, data

    async def handle_eeg(self, channel_index, data):
        timestamp = self.time_func()
        tm, d = self.unpack_eeg(data)

        if self.last_tm is None:
            self.last_tm = tm
            return

        self.data[channel_index] = d
        self.timestamps[channel_index] = timestamp

        if channel_index == 3:  # TP10 is the last in the sequence
            if tm != self.last_tm + 1 and tm != 0 and self.last_tm != 65535:
                print(f"Missing sample {tm} : {self.last_tm}")
                self.data = np.zeros((4, 12))
            else:
                self.last_tm = tm
                idxs = np.arange(12) + self.sample_index
                self.sample_index += 12
                timestamps = self.reg_params[1] * idxs + self.reg_params[0]
                self.callback_eeg(timestamps, np.transpose(self.data))
                self.data = np.zeros((4, 12))


async def main(muse):
    try:
        await muse.connect_loop()
    except asyncio.CancelledError:
        print("Async task cancelled.")
    except Exception as e:
        print(f"Main loop error: {e}")
    finally:
        print("Shutting down MuseBleak...")
        await muse.stop()


def force_disconnect(address):
    try:
        print(f"Forcing disconnect of {address} using bluetoothctl...")
        subprocess.run(["bluetoothctl", "disconnect", address], check=True)
    except Exception as e:
        print(f"bluetoothctl disconnect failed: {e}")


class AsyncRunnerThread:
    def __init__(self, coro_func, *args):
        self.coro_func = coro_func
        self.args = args
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.task = None

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.task = self.loop.create_task(self.coro_func(*self.args))
        try:
            self.loop.run_forever()
        finally:
            self._shutdown_loop()

    def _shutdown_loop(self):
        # Gracefully cancel the main task
        if self.task:
            self.loop.run_until_complete(self._cancel_task(self.task))
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        self.loop.close()

    async def _cancel_task(self, task):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            print("Main async task cancelled.")

    def start(self):
        self.thread.start()

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
