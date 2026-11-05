import numpy as np

def hutchinson_trace(matvec, n, m, dist, seed):
    rng = np.random.default_rng(seed)
    if dist == "rademacher":
        V = 2.0 * rng.integers(0, 2, size=(n, m)) - 1.0
    elif dist == "gaussian":
        V = rng.standard_normal((n, m))
    else:
        raise ValueError
    
    S = 0
    for j in range(m):
        vj = V[:, j]
        S += vj.T @ matvec(vj)
    
    return S / m

def frobenius_sq_est(matvec, n, m, seed):
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, m))

    S = [np.linalg.norm(matvec(V[:, j])) ** 2 for j in range(m)]
    return sum(S) / m

def probe_variance(A, dist):
    if dist == "rademacher":
        return 2.0 * (np.sum(A**2) - np.sum(np.diag(A)**2))
    elif dist == "gaussian":
        return 2.0 * np.trace(A @ A)
    else:
        raise ValueError

