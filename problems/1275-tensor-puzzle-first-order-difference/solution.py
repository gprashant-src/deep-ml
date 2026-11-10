import numpy as np

def diff(a: np.ndarray) -> np.ndarray:
    """out[0]=a[0]; out[i]=a[i]-a[i-1] for i>0."""
    # Your code here
    out = a.copy()
    out[1:] -= out[:-1]
    return out