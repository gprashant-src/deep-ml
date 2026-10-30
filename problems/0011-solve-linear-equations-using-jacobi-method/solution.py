import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
	a_ii = np.diag(A)
	x = np.zeros(A.shape[1])

	for _ in range(n):
		x_new = (b - (A @ x - a_ii * x)) / a_ii
		x = x_new
		
	return x