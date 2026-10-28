from typing import Callable
import numpy as np

def newtons_method_optimization(
	gradient_func: Callable[[list[float]], list[float]],
	hessian_func: Callable[[list[float]], list[list[float]]],
	x0: list[float],
	tol: float = 1e-6,
	max_iter: int = 100
) -> list[float]:
	"""
	Find the minimum of a function using Newton's method.
	
	Args:
		gradient_func: Function that returns gradient vector at a point
		hessian_func: Function that returns Hessian matrix at a point
		x0: Initial guess (list of coordinates)
		tol: Convergence tolerance for gradient norm
		max_iter: Maximum number of iterations
		
	Returns:
		The point that minimizes the function
	"""
	def G(x):
		return np.array(gradient_func(x))

	def H(x):
		return np.array(hessian_func(x))

	# Your code here
	for _ in range(max_iter):
		x_prev = np.array(x0)

		g = G(x0)
		h = H(x0)

		x1 = (x0 - (np.linalg.inv(h) @ g)).tolist()
		if np.linalg.norm(G(x1)) < tol: return x1

		x0 = x1
	
	return x0
