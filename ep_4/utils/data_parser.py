
import numpy as np

def extract_trials(annotations, cue_start, trial_length_sec):
    trials = []
    for _, row in annotations.iterrows():
        if cue_start in row['Cue']:
            start_time = row['Timestamp']
            end_time = start_time + trial_length_sec
            trials.append((start_time, end_time))
    return trials

def get_trial_data(timestamps, filtered_data, start_time, end_time, trial_length_samples):
    start_idx = np.searchsorted(timestamps, start_time)
    end_idx = start_idx + trial_length_samples
    return filtered_data[start_idx:end_idx, :]