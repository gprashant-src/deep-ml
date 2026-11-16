import numpy as np

def polynomial_basis(s, n):
    # return the (n+1)**k polynomial features of state s
    base = n + 1
    k = len(s)
    ans = []

    for num in range(base ** k):
        mask = np.base_repr(num, base=base).zfill(k)
        EXP = np.array([int(c) for c in mask])
        ans.append(np.prod(s ** EXP))
    
    return np.array(ans, dtype=float)


def gradient_mc_poly(samples, n, alpha, n_passes):
    # return the learned weight vector
    test, _ = samples[0]

    k = len(test)
    W = np.zeros((n + 1) ** k)

    for _ in range(n_passes):
        for s, G in samples:
            x = polynomial_basis(s, n)
            v_hat = np.dot(W, x)
            W = W + alpha * (G - v_hat) * x

    return W

