import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    Y = X.T @ X
    Z = X.T @ y
    X1 = np.linalg.solve(Y, Z)

    return np.round(X1, 4).tolist()
