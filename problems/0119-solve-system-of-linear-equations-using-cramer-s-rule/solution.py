import numpy as np

def cramers_rule(A, b):
    # Your code here
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    det_A = np.linalg.det(A)
    if det_A == 0:return -1

    x = np.zeros(b.shape, dtype=float)

    for i in range(A.shape[1]):
        sub_A = A.copy()
        sub_A[:, i] = b

        det_Ai = np.linalg.det(sub_A)
        x[i] = det_Ai / det_A
    
    return x