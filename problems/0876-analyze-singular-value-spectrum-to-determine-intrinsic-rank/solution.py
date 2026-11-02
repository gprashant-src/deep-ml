import numpy as np

def suggest_rank(delta_W: np.ndarray, energy_threshold: float) -> int:
	"""
	Return the smallest rank k such that the top-k singular values of delta_W
	capture at least `energy_threshold` of the total squared-singular-value energy.
	"""
	# Your code here
	U, S, Vt = np.linalg.svd(delta_W, full_matrices=True)

	if S.size == 0 or np.all(S == 0):
        return 0

	P = np.cumsum(S ** 2)
	T = P[-1]

	return np.searchsorted(P / T, energy_threshold, side="left") + 1



