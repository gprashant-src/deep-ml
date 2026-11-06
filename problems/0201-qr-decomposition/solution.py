import numpy as np

def gram_schmidt(A:np.ndarray):
    Q = np.array(A, dtype=float)
    _, n = Q.shape
    for i in range(n):
        Q[:, i] /= np.linalg.norm(Q[:, i])
        for j in range(i + 1, n):
            Q[:, j] -= np.dot(Q[:, i], Q[:, j]) * Q[:, i]
    return Q

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
	"""
	Perform QR decomposition using Gram-Schmidt process.
	
	Args:
		A: An m x n matrix represented as list of lists
	
	Returns:
		Tuple of (Q, R) where Q is orthogonal and R is upper triangular
	"""
	# Your code here
	A = np.array(A, dtype=float)
	
	Q = gram_schmidt(A)

	R = Q.T @ A

	return Q, R