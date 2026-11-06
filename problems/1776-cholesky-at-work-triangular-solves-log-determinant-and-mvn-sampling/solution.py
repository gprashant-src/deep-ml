import numpy as np

def forward_sub(L, b):
    y = np.zeros(b.shape, dtype=float)

    for i in range(len(L)):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    
    return y

def back_sub(U, y):
    x = np.zeros(y.shape, dtype=float)

    for i in range(len(U) - 1, -1, -1):
        x[i] = (y[i] - np.dot(x[i + 1:], U[i, i + 1:])) / U[i, i]

    return x

def chol_solve(A, b):
    L = np.linalg.cholesky(A)
    y = forward_sub(L, b)
    x = back_sub(L.T, y)
    return x

def logdet_chol(A):
    L = np.linalg.cholesky(A)
    return 2.0 * np.sum(np.log(np.diag(L)))

def mvn_sample(mu, Sigma, eps):
    L = np.linalg.cholesky(Sigma)
    return eps @ L.T + mu
