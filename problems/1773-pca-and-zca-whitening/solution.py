import numpy as np

def whitening_matrix(X, method="zca", eps=0.0):
    # return (mu, W)
    n, d = X.shape
    mu = X.mean(axis=0)

    Z = X - mu
    S = (Z.T @ Z) / n

    D, E = np.linalg.eigh(S)

    idx = np.argsort(D)[::-1]
    D, E = D[idx], E[:, idx]

    for j in range(d):
        max_idx = np.argmax(np.abs(E[:, j]))

        if E[max_idx, j] < 0:
            E[:, j] *= -1
    
    INV_DI = np.diag(1 / np.sqrt(D + eps))

    if method == "pca":
        W = INV_DI @ E.T
    elif method == "zca":
        W = E @ INV_DI @ E.T
    else:
        raise ValueError
    
    return mu, W



def whiten(X, mu, W):
    X1 = X - mu
    return X1 @ W.T
