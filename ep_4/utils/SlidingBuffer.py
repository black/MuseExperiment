
import numpy as np

class SlidingBuffer:
    
    def __init__(self, window_size: int, slide_size: int, n_channels: int):
        if slide_size <= 0 or window_size <= 0:
            raise ValueError("window_size and slide_size must be positive integers")
        if slide_size > window_size:
            raise ValueError("slide_size cannot be larger than window_size")

        self.window_size = window_size
        self.slide_size = slide_size
        self.n_channels = n_channels
        self.buffer = np.empty((0, n_channels), dtype=float)

    def add_samples(self, samples: np.ndarray):
        """
        Add new samples to the buffer.

        Parameters
        ----------
        samples : np.ndarray
            Shape (M, N) where M = number of samples, N = n_channels.

        Returns
        -------
        (bool, np.ndarray or None)
            - (True, window) if a window is completed.
            - (False, None) if not enough samples yet.
        """
        if samples.ndim != 2 or samples.shape[1] != self.n_channels:
            raise ValueError(f"Samples must have shape (M, {self.n_channels})")

        # Append samples to buffer
        self.buffer = np.vstack([self.buffer, samples])

        # Check if we can extract a window
        if len(self.buffer) >= self.window_size:
            window = self.buffer[:self.window_size].copy()
            # Remove slide_size samples from the start
            self.buffer = self.buffer[self.slide_size:]
            return True, window
        else:
            return False, None
