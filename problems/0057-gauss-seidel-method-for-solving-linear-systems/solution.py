import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
	if x_ini is None:
		x_ini = np.zeros(len(b))

	A, b, x_ini = map(np.asarray, (A, b, x_ini))

	a_ii = A.diagonal()

	for _ in range(n):
		for i in range(len(b)):
			x_ini[i] = (b[i] - A[i] @ x_ini + A[i, i] * x_ini[i]) / a_ii[i]

	return x_ini
