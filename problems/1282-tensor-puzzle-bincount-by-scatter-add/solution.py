import numpy as np

def bincount(a: np.ndarray, n_bins: int) -> np.ndarray:
    """Count occurrences of 0..n_bins-1 in integer array a."""
    # Your code here
    res = np.zeros(n_bins)
    np.add.at(res, a, 1)

    return res.astype(int)
