import numpy as np
import pandas as pd
from typing import List, Tuple

def extract_trials(
    annotations: pd.DataFrame, 
    cue_start: str, 
    trial_length_sec: float
) -> List[Tuple[float, float]]:
    """
    Extract start and end times for trials based on a cue.

    Parameters:
    - annotations: DataFrame with at least 'Timestamp' and 'Cue' columns
    - cue_start: string to match in the 'Cue' column
    - trial_length_sec: length of the trial in seconds

    Returns:
    - List of tuples (start_time, end_time)
    """
    trials = []
    for _, row in annotations.iterrows():
        if cue_start in str(row['Cue']):
            start_time = row['Timestamp']
            end_time = start_time + trial_length_sec
            trials.append((start_time, end_time))
    return trials


def get_trial_data(
    timestamps: np.ndarray, 
    filtered_data: np.ndarray, 
    start_time: float, 
    end_time: float, 
    trial_length_samples: int
) -> np.ndarray:
    """
    Extract a segment of data for a single trial.

    Parameters:
    - timestamps: 1D array of timestamps
    - filtered_data: 2D array of shape (samples, channels)
    - start_time: trial start time
    - end_time: trial end time
    - trial_length_samples: number of samples to extract

    Returns:
    - 2D array of shape (trial_length_samples, channels)
    """
    start_idx = np.searchsorted(timestamps, start_time, side='left')
    end_idx = start_idx + trial_length_samples
    # Ensure we don't exceed data bounds
    end_idx = min(end_idx, filtered_data.shape[0])
    return filtered_data[start_idx:end_idx, :]
