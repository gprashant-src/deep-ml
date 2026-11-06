import numpy as np

def partitioned_inverse(M, k):
    E = M[:k, :k]
    F = M[:k, k:]
    G = M[k:, :k]
    H = M[k:, k:]

    X = np.linalg.solve(H, G)
    MH = E - F @ X

    inv_H = np.linalg.inv(H)

    E1 = np.linalg.inv(MH)
    F1 = -E1 @ F @ inv_H
    G1 = -inv_H @ G @ E1
    H1 = inv_H - inv_H @ G @ F1


    return np.vstack(
        (
            np.hstack((E1, F1)), 
            np.hstack((G1, H1))
        )
    )

def block_slogdet(M, k):
    E = M[:k, :k]
    F = M[:k, k:]
    G = M[k:, :k]
    H = M[k:, k:]

    X = np.linalg.solve(H, G)
    MH = E - F @ X

    sb1, B1 = np.linalg.slogdet(MH)
    sb2, B2 = np.linalg.slogdet(H)

    return sb1 * sb2, B1 + B2

def bordered_inverse(A_inv, b, c):
    b = np.asarray(b, dtype=float).reshape(-1, 1)
    c = float(c)

    X = A_inv @ b
    MH = c - (b.T @ X).item()
    if np.abs(MH) < 10**-12:raise ValueError

    g = 1 / MH

    E1 = A_inv + g * (X @ b.T @ A_inv)
    F1 = -X * g
    G1 = -(b.T @ A_inv) * g
    H1 = np.array([[g]])


    return np.vstack(
        (
            np.hstack((E1, F1)), 
            np.hstack((G1, H1))
        )
    )

