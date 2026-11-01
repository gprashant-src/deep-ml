import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    # Your code here
    vectors = np.array(vectors, dtype=float)

    m, n = vectors.shape

    ans = []
    for i in range(m):
        u = vectors[i]
        if i == 0:
            pass
        else:
            P = np.zeros(n)
            for j in range(i):
                P += np.dot(ans[j], u) * ans[j]
            u = u - P
        N = np.linalg.norm(u)
        if N >= tol:
            q = u / N
            ans.append(q)

    return ans


