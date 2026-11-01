
import numpy as np

def matrix_image(A):
	A = np.array(A, dtype=float)
	A = A.T
	M = A.copy()
	m, n = A.shape

	for i in range(min(m, n)):
		if M[i, i] == 0:
			non_zero_rows = np.where(M[i:, i] != 0)[0]
			if len(non_zero_rows) == 0:
				continue
			pivot_row = i + non_zero_rows[0]
			M[[i, pivot_row]] = M[[pivot_row, i]]
			A[[i, pivot_row]] = A[[pivot_row, i]]

		pivot = M[i, i]
		if pivot == 0:
			continue

		for j in range(i + 1, m):
			factor = M[j, i] / pivot
			M[j] -= factor * M[i]

	row_indices = np.where(np.any(np.abs(M) > 1e-10, axis=1))[0]
	return A[row_indices].T
