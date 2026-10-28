import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	"""
	Compute derivative of composite functions using chain rule.
	
	Args:
		functions: List of function names (applied right to left)
		          Available: 'square', 'sin', 'exp', 'log'
		x: Point at which to evaluate derivative
	
	Returns:
		Derivative value at x
	
	Example:
		['sin', 'square'] represents sin(x²)
		['exp', 'sin', 'square'] represents exp(sin(x²))
	"""
	# Your code here
	def square(x): return [x ** 2, 2 * x]
	def sin(x): return [np.sin(x), np.cos(x)]
	def exp(x): return [np.exp(x), np.exp(x)]
	def log(x): return [np.log(x), 1 / x]

	H = {
		"square": square,
		"sin": sin,
		"exp": exp,
		"log": log
	}

	y = x
	ans = 1
	for op in reversed(functions):
		yi, y_hat_i = H[op](y)
		ans = ans * y_hat_i
		y = yi
	
	return ans