import numpy as np

def ones(n: int) -> np.ndarray:
    """Return a length-n float vector of ones without calling np.ones."""
    # Your code here
    out = np.arange(1, n + 1)
    out[1:] -= out[:-1]
    return out.astype(float)

