import numpy as np

def f(A:np.ndarray):
	if len(A) == 2:
		return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
	
	res = 0
	for i in range(len(A)):
		sign = (-1) ** i
		X = np.delete(np.delete(A, 0, axis=0), i, axis=1)
		res += A[0, i] * sign * f(X)
	
	return res
def determinant_4x4(matrix: list[list[int|float]]) -> float:
	# Your recursive implementation here
	A = np.array(matrix, dtype=float)
	return f(A)	