import numpy as np

from multiprocessing import Queue, Event, Process

# Import the visualizer we defined earlier
from utils.filters import *
from utils.SlidingBuffer import *
from utils.models import *
from utils.audio import *
from utils.eeg_utils import *
from utils.transform_functions import *
from utils.visualization import audio_biofeedback_visualizer  # adjust import path

class AudioBioFeedbackEngine:
    
    ALPHA_CHANNELS = [0, 3]
    SAMPLING_RATE = 256

    def __init__(self):
        self.active = False
        self.filtering = OnlineBandpassFilterBank(1, 40, 256, 4, order=2)
        self.buffer = SlidingBuffer(window_size=512, slide_size=32, n_channels=4)
        self.model = OnlineTwoStateGMM(mu_prior=[0.10, 0.20], sigma_prior=[0.05, 0.12], pi_prior=[0.5, 0.5], eta=0.01, pi_min=0.05)
        self.audioEngine = RealTimePitchSynth()
        self.volumeHistory = []
        self.pitchHistory = []
        self.relative_alpha = []

        # Visualizer communication
        self.vis_queue = Queue()
        self.shutdown_event = Event()
        self.vis_process = None

    def start(self):
        self.active = True
        self.audioEngine.start()
        self.model.reset()

        # Start the visualizer process
        self.vis_process = Process(target=audio_biofeedback_visualizer, args=(self.vis_queue, self.shutdown_event))
        self.vis_process.start()

    def stop(self):
        self.active = False
        self.audioEngine.set_volume(0)

        # Stop visualizer
        if self.vis_process:
            self.shutdown_event.set()
            self.vis_process.join()

    def add_samples(self, samples: np.ndarray):
        if not self.active:
            return

        filteredSamples = self.filtering.process(samples)
        complete, window = self.buffer.add_samples(filteredSamples)

        if complete:
            relative_alphas = np.array([
                compute_relative_alpha_from_signal(window[:, ch], fs=AudioBioFeedbackEngine.SAMPLING_RATE, alpha_band=(8, 13))
                for ch in AudioBioFeedbackEngine.ALPHA_CHANNELS
            ])
            
            relative_alpha = relative_alphas.mean()
            self.model.update(relative_alpha)
            params = self.model.get_model_parameters()
            mu1, mu2 = params["mu"]
            sig1, sig2 = params["sigma"]

            pitchControl = parameterized_sigmoid(relative_alpha, mu1, mu2)
            volumeControl = piecewise_linear(relative_alpha, mu1, mu2)

            self.audioEngine.set_frequency(110 + 330 * pitchControl)
            self.audioEngine.set_volume(volumeControl * 0.25)

            self.pitchHistory.append(pitchControl)
            self.relative_alpha.append(relative_alpha)
            self.volumeHistory.append(volumeControl)

            # Push parameters to visualizer
            self.vis_queue.put(np.array([[relative_alpha, mu1, mu2, pitchControl, volumeControl]]))

            print(f"relative_alpha: {relative_alpha}, mu: ({mu1}, {mu2})")
