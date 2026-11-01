import numpy as np

def sinkhorn_knopp(B: list, t_max: int = 20) -> list:
    """
    Project a square matrix onto the set of doubly stochastic matrices.

    Args:
        B: n x n matrix as a list of lists (real-valued).
        t_max: number of normalization iterations.

    Returns:
        A nested list representing the resulting doubly stochastic matrix.
    """
    # Your code here
    B = np.array(B, dtype=float)

    M0 = np.exp(B)

    for _ in range(t_max):
        M0 /= M0.sum(axis=1, keepdims=True)
        M0 /= M0.sum(axis=0, keepdims=True)
    
    return M0.tolist()
