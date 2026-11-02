import numpy as np

def subspace_similarity(A: list, B: list, i: int, j: int) -> float:
    """
    Compute the normalized subspace similarity between the top-i left singular
    subspace of A and the top-j left singular subspace of B.

    Args:
        A: matrix as a list of lists
        B: matrix as a list of lists (same number of rows as A)
        i: number of top left singular vectors to take from A
        j: number of top left singular vectors to take from B

    Returns:
        A float in [0, 1] measuring subspace similarity.
    """
    # Your code here
    A, B = np.array(A, dtype=float), np.array(B, dtype=float)

    UA, SA, VA = np.linalg.svd(A)
    UB, SB, BV = np.linalg.svd(B)

    UA = UA[:, :i]
    UB = UB[:, :j]

    N = np.linalg.norm(UA.T @ UB, ord='fro')

    return N**2 / min(i, j)
