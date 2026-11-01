import numpy as np

def lu_decomposition(A: list) -> tuple:
	"""
	Perform LU decomposition on a square matrix using Doolittle's method.
	
	Args:
		A: Square matrix as a list of lists
	
	Returns:
		tuple: (L, U) where L is lower triangular with 1s on diagonal,
		       U is upper triangular, and A = L @ U
	"""
	A = np.array(A, dtype=float)

	U = A.copy()
	L = np.eye(A.shape[0])

	m, n = A.shape

	for i in range(m):
		# pivot_row = i + np.argmax(np.abs(U[i:, i]))

		# if U[pivot_row, i] == 0:
		# 	continue

		# if pivot_row != i:
		# 	A[[i, pivot_row]] = A[[pivot_row, i]]
		# 	b[[i, pivot_row]] = b[[pivot_row, i]]

		pivot = U[i, i]
		for j in range(i + 1, m):
			factor = U[j, i] / pivot
			U[j] -= factor * U[i]
			L[j][i] = factor
			# b[j] -= factor * b[i]

	return L, U
	