import numpy as np

def kron_matvec(A, B, C):
    # vec_C = C.reshape(-1, order="F")
    res = B @ C @ A.T
    return res.reshape(-1, order="F")

def kron_solve(A, B, y):
    l = len(y)
    q = B.shape[0]
    n = l // q

    y = y.reshape(q, n, order="F")
    x1 = np.linalg.solve(B, y)
    
    x2 = np.linalg.solve(A, x1.T)
    return x2.T.reshape(-1, order="F")


def bow_logits(S, W, V):
    return np.einsum('ntk,kd,dc->nc', S, W, V, optimize=True)
