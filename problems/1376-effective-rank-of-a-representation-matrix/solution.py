import numpy as np

def effective_rank(H, tol=1e-12):
    """
    Exponentiated Shannon entropy of the normalized singular values.

    Args:
        H: 2D array
        tol: singular values below this are discarded

    Returns:
        float in [1, number of retained singular values]
    """
    # Your code here
    S = np.linalg.svd(H, compute_uv=False)

    P = S / S.sum()

    ES = -(P * np.log(P)).sum()
    return float(np.exp(ES))

