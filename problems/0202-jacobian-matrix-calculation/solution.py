import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Jacobian matrix using numerical differentiation.
	
	Args:
		f: Function that takes a list and returns a list
		x: Point at which to evaluate the Jacobian
		h: Step size for finite differences
	
	Returns:
		Jacobian matrix as list of lists
	"""
	# Your code here
	y = np.array(f(x))

	J = np.zeros((len(y), len(x)))
	for j in range(len(x)):
		x_temp = x.copy()
		x_temp[j] += h

		y_h = np.array(f(x_temp), dtype=float)
		J[:, j] = (y_h - y) / h

	return J.tolist()