import numpy as np

def smote(X_minority: np.ndarray, n_synthetic: int, k: int = 5) -> np.ndarray:
    """
    Generate synthetic samples using SMOTE algorithm.

    Note: the random seed is set by the grader before your function runs,
    so you do NOT need to set it. Just use numpy's global RNG directly
    (np.random.randint, np.random.random, ...).

    Args:
        X_minority: 2D array of minority class samples (n_samples, n_features)
        n_synthetic: Number of synthetic samples to generate
        k: Number of nearest neighbors to consider

    Returns:
        2D array of synthetic samples (n_synthetic, n_features)
    """
    # Your code here
    n_samples, n_features = X_minority.shape
    k_actual = min(k, n_samples - 1)

    if k_actual == 0 or n_synthetic == 0:return np.empty((0, n_features))

    ans = np.empty((n_synthetic, n_features))

    for k in range(n_synthetic):
        i = np.random.randint(0, n_samples)
        j = np.random.randint(0, k_actual)
        gap = np.random.random()

        dist = np.linalg.norm(X_minority - X_minority[i], axis=1)
        idx = np.argsort(dist)

        x_nn = X_minority[idx[j + 1]]
        ans[k] = X_minority[i] + gap * (x_nn - X_minority[i])
    
    return ans
