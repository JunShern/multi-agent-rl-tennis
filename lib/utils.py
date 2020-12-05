import numpy as np

def moving_averages(values, window=100):
    return [np.mean(values[:i+1][-window:]) for i, _ in enumerate(values)]