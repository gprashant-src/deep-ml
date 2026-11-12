import numpy as np

def min_norm_solution(A, b):
    M = (A @ A.T)
    return A.T @ np.linalg.solve(M, b)

def min_norm_via_kkt(A, b):
    # return (x, lam)
    lam = np.linalg.solve(A @ A.T, -2 * b)
    x = -0.5 * (A.T @ lam)
    return x, lam

def min_norm_from_particular(A, x_p):
    rhs = A @ x_p
    y = np.linalg.solve(A @ A.T, rhs)
    return A.T @ y
