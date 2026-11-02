import numpy as np

def grassmann_distance(U_A: list[list[float]], U_B: list[list[float]]) -> float:
	"""
	Compute the projection-based Grassmann distance between two subspaces
	represented by column-orthonormal matrices U_A and U_B.
	"""
	U_A = np.array(U_A, dtype=float)
	U_B = np.array(U_B, dtype=float)

	d, i = U_A.shape
	d, j = U_B.shape

	p = min(i, j)

	U, S, vt = np.linalg.svd(U_A.T @ U_B, full_matrices=True)

	Sp = np.clip(S[:p], 0, 1)

	return np.sqrt(np.max(p - (Sp ** 2).sum(), 0))

