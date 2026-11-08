import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
	"""
	Solve the system Ax = b using the Conjugate Gradient method.

	:param A: Symmetric positive-definite matrix
	:param b: Right-hand side vector
	:param n: Maximum number of iterations
	:param x0: Initial guess for solution (default is zero vector)
	:param tol: Convergence tolerance
	:return: Solution vector x
	"""
	if x0 is None:
		x0 = np.zeros(b.shape)
	
	r0 = b - A @ x0
	p0 = r0.copy()

	for _ in range(n):
		alpha_0 = (r0.T @ r0) / (p0.T @ A @ p0)
		x1 = x0 + alpha_0 * p0

		x0 = x1
		r1 = r0 - alpha_0 * (A @ p0)

		if np.linalg.norm(r1) < tol:
			break
		
		beta_k = (r1.T @ r1) / (r0.T @ r0)

		p1 = r1 + beta_k * p0

		r0, p0 = r1, p1
	
	return x0
	
