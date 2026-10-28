import numpy as np

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
	"""
	Compute partial derivatives of multivariable functions.
	
	Args:
		func_name: Function identifier
			'poly2d': f(x,y) = x²y + xy²
			'exp_sum': f(x,y) = e^(x+y)
			'product_sin': f(x,y) = x·sin(y)
			'poly3d': f(x,y,z) = x²y + yz²
			'squared_error': f(x,y) = (x-y)²
		point: Point (x, y) or (x, y, z) at which to evaluate
	
	Returns:
		Tuple of partial derivatives (∂f/∂x, ∂f/∂y, ...) at point
	"""
	# Your code here
	H = {
		"poly2d": lambda x, y: [2*x*y + y**2, 2*x*y + x**2, 0],
		"exp_sum": lambda x, y: [np.exp(x + y), np.exp(x + y), 0],
		"product_sin": lambda x, y: [np.sin(y), x*np.cos(y), 0],
		"poly3d": lambda x, y, z: [2*x*y, x**2 + z**2, 2*y*z],
		"squared_error": lambda x, y: [2 * (x - y), -2 * (x - y), 0]
	}

	ans = H[func_name](*point)
	return ans if len(ans) == 3 else ans[:2]