import numpy as np

def rotation_layer(X, angle):
    A = [[np.cos(angle), -np.sin(angle)], 
         [np.sin(angle), np.cos(angle)]]

    A = np.array(A, dtype=float)
    X = np.array(X, dtype=float)

    return (X @ A.T).tolist()