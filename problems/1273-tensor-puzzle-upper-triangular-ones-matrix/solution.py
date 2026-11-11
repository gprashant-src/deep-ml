import numpy as np

def triu_ones(n: int) -> np.ndarray:
    """n x n upper-triangular matrix of ones (including diagonal)."""
    # Your code here
    row, col = np.arange(n)[:, None], np.arange(n)

    return np.where(row <= col, 1, 0).astype(float)
