import numpy as np

def power_iteration(A, v0, n_iter):
    # return (lam, u): Rayleigh quotient and unit vector, largest-|entry| positive
    v = v0 / np.linalg.norm(v0)

    for _ in range(n_iter):
        Y = A @ v
        v = Y / np.linalg.norm(Y)

    u = v

    max_idx = np.argmax(np.abs(u))
    if u[max_idx] < 0:
        u = -u
    
    return (u.T @ A @ u) / np.dot(u, u), u
    

def deflate(A, lam, u):
    return A - lam * np.outer(u, u)

def top_k_eigs(A, k, n_iter, v0):
    # return (vals (k,), vecs (n, k))
    A0 = A.copy()
    n, _ = A.shape

    eig_vals, eig_vecs = np.zeros(k), np.zeros((n, k))

    for i in range(k):
        X, Y = power_iteration(A0, v0, n_iter)
        eig_vals[i] = X
        eig_vecs[:, i] = Y

        A0 = deflate(A0, X, Y)
    
    return eig_vals, eig_vecs

    
