import numpy as np
from scipy.optimize import linprog

def lad_fit(X, y):
    N, D = X.shape
    c = np.hstack([np.zeros(D), np.ones(N)])
    A_ub = np.vstack(
        [np.hstack([X, -np.eye(N)]), np.hstack([-X, -np.eye(N)])]
    )
    b_ub = np.hstack([y, -y])
    bounds = [(None, None)] * D + [(0, None)] * N
    return linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs").x[:D]

def huber_fit(X, y, delta, n_iter):
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(n_iter):
        r = y - X @ w
        s = np.where(r == 0, 1.0, np.minimum(1.0, delta / np.abs(r)))
        sqrt_s = np.sqrt(s)
        w = np.linalg.lstsq(sqrt_s[:, None] * X, sqrt_s * y, rcond=None)[0]
    return w
