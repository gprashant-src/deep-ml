import numpy as np

def projection_amplification(W: list[list[float]], delta_W: list[list[float]], r: int) -> float:
	"""
	Compute the amplification factor of delta_W with respect to W
	using the top-r singular subspaces of delta_W.

	Returns: ||delta_W||_F / ||U_r^T W V_r||_F
	"""
	W = np.array(W, dtype=float)
	delta_W = np.array(delta_W, dtype=float)


	U, S, Vt = np.linalg.svd(delta_W, full_matrices=True)

	Ur, Vtr = U[:, :r], Vt[:r, :].T

	W_del_hat = Ur.T @ W @ Vtr

	N_W_hat = np.linalg.norm(W_del_hat, ord='fro')
	N_del_W = np.linalg.norm(delta_W, ord='fro')

	return N_del_W / N_W_hat
