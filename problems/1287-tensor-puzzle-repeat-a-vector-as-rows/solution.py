import numpy as np

def repeat_rows(a: np.ndarray, d: int) -> np.ndarray:
    """Stack d copies of 1-D array a as rows."""
    # Your code here
    n = len(a)
    ans = np.ones((d, n))

    ans = ans * a
    return ans
