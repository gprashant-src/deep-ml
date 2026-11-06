import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    # Your code here
    a1, a2 = A[:, 0], A[:, 1]

    alpha = np.dot(a1, a1)
    beta = np.dot(a2, a2)
    gamma = np.dot(a1, a2)

    tau = (beta - alpha) / (2 * gamma)

    theta = 0.5 * np.arctan(1 / tau)

    c = np.cos(theta)
    s = np.sin(theta)

    J = np.array([[c, s], [-s, c]])

    B = A @ J

    E = np.sqrt(np.diag(B.T @ B))
    idx = np.argsort(E)[::-1]

    sigma = E[idx]
    B = B[:, idx]
    U = B / sigma[None, :]
    Vt = J[:, idx]
    
    print(U @ np.diag(sigma) @ Vt)
    return U, sigma, Vt



