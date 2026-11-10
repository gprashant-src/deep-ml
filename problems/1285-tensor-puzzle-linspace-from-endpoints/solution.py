import numpy as np

def linspace(start, stop, n: int) -> np.ndarray:
    """n evenly spaced values from start to stop inclusive."""
    # Your code here
    if n == 1:
        return np.array([start])
    step = (stop - start) / (n - 1)
    return np.array([start + i * step for i in range(n)])
