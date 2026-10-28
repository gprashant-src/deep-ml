import numpy as np

def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	"""
	Compute the derivative of cross-entropy loss with respect to logits.
	
	Args:
		logits: Raw model outputs (before softmax)
		target: Index of the true class (0-indexed)
		
	Returns:
		Gradient vector where gradient[i] = dL/d(logits[i])
	"""
	# Your code here
	E = np.exp(logits)

	n = len(logits)

	Y = np.array([1 if i == target else 0 for i in range(n)])

	soft = E / E.sum()

	return (soft - Y).tolist()