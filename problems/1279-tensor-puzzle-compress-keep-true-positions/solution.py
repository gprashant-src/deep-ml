import numpy as np

def compress(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Pack v[g] into the front of a zero vector of length len(v)."""
    # Your code here
    idx = np.where(g == True)[0]

    VC = v[idx]

    VC = np.pad(VC, (0, len(g) - len(VC)))
    return VC


