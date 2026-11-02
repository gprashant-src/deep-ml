import numpy as np

def newton_schulz(M, num_iters: int, a: float, b: float, c: float):
    """
    Apply Newton-Schulz iterations to approximately orthogonalize M.
    Returns the resulting matrix as a nested list of floats.
    """
    # Your code here
    M = np.array(M, dtype=float)
    norm = np.linalg.norm(M, ord='fro')
    if norm == 0:
        return M.tolist()

    M = M / norm

    for _ in range(num_iters):
        X = M @ M.T
        Y = X @ X

        M = a * M + (b * X @ M) + (c * Y @ M)
    
    return M.tolist()


