import numpy as np

def row_normalize(counts: list[list[float]]) -> list[list[float]]:
    """Convert a count matrix into a row-stochastic probability matrix."""
    counts = np.array(counts, dtype=float)
    
    ans = counts / counts.sum(axis=1, keepdims=True)
    return np.where(np.isnan(ans), 0, ans).tolist()

