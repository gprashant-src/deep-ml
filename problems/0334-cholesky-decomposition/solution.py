import numpy as np

def check(A:np.ndarray):
    if A.size == 0: return False

    m, n = A.shape
    if m != n:return False

    E = np.linalg.eigvals(A).real
    return bool(np.allclose(A, A.T) and np.all(E > 0))

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.
    
    Args:
        A: A symmetric positive-definite matrix (2D list or numpy array)
    
    Returns:
        L: Lower triangular matrix such that A = L @ L.T as a 2D list,
           or -1 if decomposition is not possible
    """
    # Your code here
    A = np.array(A, dtype=float)

    if not check(A): return -1

    m = A.shape[0]
    L = np.zeros((m, m), dtype=float)

    for j in range(m):
        L[j, j] = np.sqrt(A[j, j] - L[j, :j] @ L[j, :j])
        L[j + 1:, j] = (A[j + 1:, j] - L[j + 1:, :j] @ L[j, :j]) / L[j, j]
    
    return L.tolist()