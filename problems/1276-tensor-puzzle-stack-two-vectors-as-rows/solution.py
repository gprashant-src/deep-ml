import numpy as np

def vstack(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Stack 1-D arrays a and b as rows of a (2, n) matrix."""
    # Your code here
    r = np.arange(2)[:, None]

    mask = (r == 0).astype(float)

    return mask * a + (1 - mask) * b
