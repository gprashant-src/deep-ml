import numpy as np

def heaviside(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise heaviside with zero-value b."""
    # Your code here
    out = b.copy()

    out[a < 0] = 0
    out[a == 0] = b[a == 0]
    out[a > 0] = 1

    return out
