import numpy as np

def diag(A: np.ndarray) -> np.ndarray:
    """Return the main diagonal of square matrix A."""
    # Your code here
    X = A.ravel()

    m,n = A.shape
    idx = np.arange(0, A.size, n + 1)

    return X[idx]
