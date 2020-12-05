import numpy as np

def moving_averages(values, window=100):
    """
    Input: A 1D list of N values
    Return: A 1D list of N values corresponding to the average of
    the last [window] values from that same index in the input.
    Where the current index is <= window, the average is taken
    from the start of the list.
    """
    values = np.array(values)
    assert len(values.shape) == 1 # Values should be a 1D list
    return [np.mean(values[:i+1][-window:]) for i, _ in enumerate(values)]