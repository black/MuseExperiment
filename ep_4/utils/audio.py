
import numpy as np
import sounddevice as sd
import threading

def soft_beep(frequency=440, duration=0.5, volume=0.2, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = volume * np.sin(2 * np.pi * frequency * t)
    sd.play(wave, samplerate=sample_rate)
    sd.wait()
    
class RealTimePitchSynth:
    def __init__(self, initial_freq=440.0, amplitude=0.3, sample_rate=44100):
        self.freq = initial_freq
        self.amplitude = amplitude
        self.sample_rate = sample_rate
        self.running = False
        self._lock = threading.Lock()
        self.phase = 0.0  # Track phase between callbacks

    def _audio_callback(self, outdata, frames, time_info, status):
        with self._lock:
            freq = self.freq
            amp = self.amplitude

        phase_increment = 2 * np.pi * freq / self.sample_rate
        phase_array = self.phase + phase_increment * np.arange(frames)
        wave = amp * np.sin(phase_array)
        self.phase = (phase_array[-1] + phase_increment) % (2 * np.pi)  # wrap phase

        outdata[:] = wave.reshape(-1, 1)

    def start(self):
        if self.running:
            print("Synth already running.")
            return
        self.running = True
        self.stream = sd.OutputStream(callback=self._audio_callback,
                                      samplerate=self.sample_rate,
                                      channels=1)
        self.stream.start()
        print("Synth started.")

    def stop(self):
        if self.running:
            self.stream.stop()
            self.stream.close()
            self.running = False
            print("Synth stopped.")

    def set_frequency(self, new_freq):
        with self._lock:
            self.freq = float(new_freq)

    def set_volume(self, new_amp):
        """Set amplitude (0.0 = silent, 1.0 = full scale)."""
        if not (0.0 <= new_amp <= 1.0):
            raise ValueError("Volume must be between 0.0 and 1.0")
        with self._lock:
            self.amplitude = float(new_amp)

    def interactive_control(self):
        print("Type a new frequency (e.g. 880), 'v <value>' to set volume, or 'q' to quit.")
        try:
            while True:
                cmd = input("> ").strip()
                if cmd.lower() == 'q':
                    break
                elif cmd.startswith("v "):
                    try:
                        new_amp = float(cmd.split()[1])
                        self.set_volume(new_amp)
                        print(f"Volume set to {new_amp:.2f}")
                    except (ValueError, IndexError):
                        print("Usage: v <volume between 0.0 and 1.0>")
                else:
                    try:
                        self.set_frequency(float(cmd))
                        print(f"Frequency set to {cmd} Hz")
                    except ValueError:
                        print("Invalid input.")
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    
# Example usage:
if __name__ == "__main__":
    synth = RealTimePitchSynth()
    synth.start()
    synth.interactive_control()