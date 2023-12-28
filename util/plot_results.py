import numpy as np
from typing import List
import matplotlib.pyplot as plt

def plot_MA_log10(numbers: List, window: int, plot_name: str, label = ""):

    plt.figure(figsize=(10, 6))

    moving_avg = np.convolve(np.log10(numbers), np.ones(window) / window, mode='valid')
    plt.plot(moving_avg)
    plt.grid(True)
    plt.legend()
    plt.title(label)
    plt.savefig(plot_name)
