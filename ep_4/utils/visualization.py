"""
Real-Time EEG Visualizer for Muse Headset (Lesson 1)

Author: Fred Simard, RE-AK Technologies, 2025
Website: https://www.re-ak.com
Discord: https://discord.gg/XzeDHJf6ne

Description:
------------
This module implements a real-time EEG visualizer using PyQtGraph and multiprocessing.

It receives EEG samples from a queue and plots the four Muse channels (TP9, AF7, AF8, TP10)
with vertical offsets for clarity. It also applies a live bandpass filter to reduce noise.

This script is intended for educational/demo purposes and is part of Lesson 1 in the Muse
EEG YouTube tutorial series, demonstrating EEG artifacts such as eye blinks, jaw clenches,
and movement-induced noise.

Main Components:
----------------
- `visualizer(queue: Queue, shutdown_event: Event)`:
    Launches a PyQtGraph-based GUI window in a separate process. Handles real-time EEG data
    visualization and graceful shutdown triggered via multiprocessing event.
"""

import numpy as np
from multiprocessing import Queue, Event

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets, QtCore



def audio_biofeedback_visualizer(queue: Queue, shutdown_event: Event):
    """
    Real-time Audio Biofeedback parameter plotter.

    Parameters:
    -----------
    queue : multiprocessing.Queue
        Queue through which raw parameter samples (shape: [samples, 5]) are received.
        Columns: [sample_value, mu1, mu2, pitch_control, volume_control]

    shutdown_event : multiprocessing.Event
        Event used to signal that the visualizer should terminate cleanly.
    """

    app = QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(title="Audio Biofeedback Parameters")
    win.resize(1000, 500)
    win.show()

    plot = win.addPlot(title="Audio Biofeedback Parameters")
    plot.showGrid(x=True, y=True)
    plot.setLabel('left', 'Value')
    plot.setLabel('bottom', 'Time (s)')
    plot.setYRange(0, 1)  # Values between 0 and 1

    # Colors for the lines
    colors = [pg.mkPen('r', width=2),  # mu1
              pg.mkPen('g', width=2),  # mu2
              pg.mkPen('b', width=2),  # pitch_control
              pg.mkPen('y', width=2)]  # volume_control

    # Add legend (top-left by default)
    legend = plot.addLegend(offset=(10, 10))
    
    # Lines for other parameters, each with a label
    param_names = ["mu1", "mu2", "pitch_control", "volume_control"]
    lines = [plot.plot(pen=colors[i], name=param_names[i]) for i in range(4)]

    # Scatter plot for sample_value
    scatter = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120), name="Sample Value")
    plot.addItem(scatter)

    # Lines for other parameters
    lines = [plot.plot(pen=colors[i]) for i in range(4)]

    sampling_rate = 8  # Hz
    history_seconds = 10
    buffer_size = sampling_rate * history_seconds

    # Buffers
    sample_buffer = np.zeros(buffer_size)
    param_buffers = [np.zeros(buffer_size) for _ in range(4)]
    time_buffer = np.linspace(-history_seconds, 0, buffer_size)

    def update():
        while not queue.empty():
            samples = queue.get()  # shape: [n_samples, 5]
            for sample in samples:
                # Shift buffers
                sample_buffer[:-1] = sample_buffer[1:]
                sample_buffer[-1] = sample[0]  # sample_value

                for i in range(4):
                    param_buffers[i][:-1] = param_buffers[i][1:]
                    param_buffers[i][-1] = sample[i+1]

        # Update scatter
        scatter.setData(x=time_buffer, y=sample_buffer)

        # Update lines
        for i in range(4):
            lines[i].setData(time_buffer, param_buffers[i])

        if shutdown_event.is_set():
            print("[VISUALIZER] Shutdown signal received. Closing app...")
            app.quit()

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(int(1000 / sampling_rate))  # update roughly every sample interval

    app.exec_()