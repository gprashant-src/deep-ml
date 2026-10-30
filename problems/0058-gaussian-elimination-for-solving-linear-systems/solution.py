import numpy as np

def gaussian_elimination(A, b):
	m, n = A.shape

	for i in range(m):
		pivot_row = i + np.argmax(np.abs(A[i:, i]))

		if A[pivot_row, i] == 0:
			continue

		if pivot_row != i:
			A[[i, pivot_row]] = A[[pivot_row, i]]
			b[[i, pivot_row]] = b[[pivot_row, i]]

		pivot = A[i, i]
		for j in range(i + 1, m):
			factor = A[j, i] / pivot
			A[j] -= factor * A[i]
			b[j] -= factor * b[i]

	x = np.zeros(n)

	for i in range(n - 1, -1, -1):
		x[i] = (b[i] - A[i, i+1:] @ x[i+1:]) / A[i, i]

	return x
