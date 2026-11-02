import numpy as np

def cumsum(a: np.ndarray) -> np.ndarray:
    """Inclusive cumulative sum of 1-D array a without np.cumsum."""
    # Your code here
    n = len(a)

    L = np.tril(np.ones((n, n), dtype=float))

    return L @ a
