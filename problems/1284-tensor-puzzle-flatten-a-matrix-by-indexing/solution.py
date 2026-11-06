import numpy as np

def flatten(A: np.ndarray) -> np.ndarray:
    """Row-major flatten of 2-D array A without reshape/ravel."""
    # Your code here
    rows, cols = A.shape

    res = np.zeros(rows*cols, dtype=float)

    j = 0
    for i in range(rows):
        res[j:j + cols] = A[i]
        j += cols
    return res
