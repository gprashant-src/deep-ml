import numpy as np

def woodbury_solve(sigma_diag, X, B):
    N, D = X.shape

    INV = np.diag(1 / sigma_diag)

    C = np.linalg.inv(np.eye(D) + (X.T @ INV @ X))
    Y = INV @ X

    return (INV - Y @ C @ Y.T) @ B


def sherman_morrison(A_inv, u, v):
    u = np.asarray(u).reshape(-1, 1)
    v = np.asarray(v).reshape(-1, 1)
    
    den = (1 + v.T @ A_inv @ u).item()
    if den == 0:raise ValueError

    num = A_inv @ (u @ v.T) @ A_inv
    return A_inv - num / den

def logdet_rank_one_update(logdet_A, A_inv, u, v):
    u = np.asarray(u).reshape(-1, 1)
    v = np.asarray(v).reshape(-1, 1)

    S = (1 + (v.T @ A_inv @ u).item())
    if S <= 0:raise ValueError

    return float(logdet_A + np.log(S))
