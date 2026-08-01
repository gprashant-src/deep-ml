import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	if all(d == 0 for d in gradient):return {"magnitude": 0, "direction": gradient, "descent_direction": gradient}

	A = np.sqrt(sum(d ** 2 for d in gradient))
	B = [d / A for d in gradient]
	C = [-1 * v for v in B]
	return {"magnitude": A, "direction": B, "descent_direction": C}
	