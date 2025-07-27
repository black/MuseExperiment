

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