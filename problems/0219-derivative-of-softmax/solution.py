import numpy as np

def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	# Your code here
	x = np.array(x)
	n = len(x)
	
	ans = [[0] * n for _ in range(n)]

	E = np.exp(x)

	soft = E / E.sum()
	
	for i in range(n):
		for j in range(i, n):
			if i == j:
				ans[i][j] = soft[i] * (1 - soft[i])
			else:
				ans[i][j] = ans[j][i] = -soft[i] * soft[j]
	
	return ans
	