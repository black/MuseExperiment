
import numpy as np
from scipy.signal import butter, lfilter, filtfilt


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a Butterworth bandpass filter to the signal.

    Parameters:
        data (ndarray): Input signal (1D).
        lowcut (float): Low cutoff frequency (Hz).
        highcut (float): High cutoff frequency (Hz).
        fs (int): Sampling rate (Hz).
        order (int): Filter order.

    Returns:
        ndarray: Filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)
    
    
    
    

class OnlineBandpassFilterBank:
    def __init__(self, lowcut, highcut, fs, n_channels, order=4):
        """
        Initialize a filter bank with one bandpass filter per channel.

        Parameters
        ----------
        lowcut : float
            Low cutoff frequency (Hz).
        highcut : float
            High cutoff frequency (Hz).
        fs : int
            Sampling rate (Hz).
        n_channels : int
            Number of channels.
        order : int, optional
            Filter order (default=4).
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        self.b, self.a = butter(order, [low, high], btype='band')
        self.n_channels = n_channels

        # Initialize filter state for each channel
        self.zi = [np.zeros(max(len(self.a), len(self.b)) - 1) for _ in range(n_channels)]

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Process incoming multichannel samples.

        Parameters
        ----------
        samples : np.ndarray
            Shape (M, N), M = number of samples, N = n_channels.

        Returns
        -------
        np.ndarray
            Filtered samples, same shape (M, N).
        """
        if samples.ndim != 2 or samples.shape[1] != self.n_channels:
            raise ValueError(f"Samples must have shape (M, {self.n_channels})")

        filtered = np.zeros_like(samples, dtype=float)

        for ch in range(self.n_channels):
            filtered[:, ch], self.zi[ch] = lfilter(self.b, self.a, samples[:, ch], zi=self.zi[ch])

        return filtered
