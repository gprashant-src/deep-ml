import numpy as np

def pad_to(a: np.ndarray, j: int) -> np.ndarray:
    """Pad a with zeros (or truncate) to length j."""
    # Your code here
    n = len(a)

    if j < len(a):
        return a[:j]
    else:
        out = np.zeros(j)
        out[:len(a)] = a
        return out
