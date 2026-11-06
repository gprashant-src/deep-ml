import numpy as np

def flatten(A: np.ndarray) -> np.ndarray:
    """Row-major flatten of 2-D array A without reshape/ravel."""
    # Your code here
    rows, cols = A.shape
    total = rows * cols

    idx = np.arange(total)

    row_idx = idx // cols
    col_idx = idx % cols

    return A[row_idx, col_idx]

