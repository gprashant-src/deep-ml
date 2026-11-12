import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
    A = np.array(A, dtype=float)
    T = np.array(T, dtype=float)
    S = np.array(S, dtype=float)

    if T.shape[0] != T.shape[1] or np.linalg.det(T) == 0 or S.shape[0] != S.shape[1] or np.linalg.det(S) == 0:
        return -1
    
    X = np.linalg.solve(T, A)
    return (X @ S).tolist()