import numpy as np

def eye(n: int) -> np.ndarray:
    """n x n identity matrix without np.eye."""
    # Your code here
    row, col = np.arange(n)[:, None], np.arange(n)

    return np.where(row == col, 1, 0).astype(float)
