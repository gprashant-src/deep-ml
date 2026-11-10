import numpy as np

def roll(a: np.ndarray) -> np.ndarray:
    """Circular left shift by one."""
    # Your code here
    n = len(a)
    idx = np.arange(n)
    return a[(idx + 1) % n]
