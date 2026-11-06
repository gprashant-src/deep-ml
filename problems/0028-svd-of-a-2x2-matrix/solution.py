import numpy as np

def gram_schmidt(A:np.ndarray):
    Q = np.array(A, dtype=float)
    _, n = Q.shape
    for i in range(n):
        Q[:, i] /= np.linalg.norm(Q[:, i])
        for j in range(i + 1, n):
            Q[:, j] -= np.dot(Q[:, i], Q[:, j]) * Q[:, i]
    return Q

def svd_2x2(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix.
    
    Args:
        A: 2x2 numpy array
    
    Returns:
        U: 2x2 orthogonal matrix (left singular vectors)
        s: 1D array of singular values
        V: 2x2 matrix (right singular vectors)
    """
    # Your code here
    E, V = np.linalg.eigh(A.T @ A)

    idx = np.argsort(E)[::-1]
    E = E[idx]
    V = V[:, idx]

    V = gram_schmidt(V)

    sigma = np.sqrt(E)
    sigma_inv = 1.0 / sigma

    U = A @ V @ np.diag(sigma_inv)

    return U, sigma, V.T
    