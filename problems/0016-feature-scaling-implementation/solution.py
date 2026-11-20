import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    # Your code here
    mu, std = data.mean(axis=0, keepdims=True), data.std(axis=0, keepdims=True)

    standardized_data = (data - mu) / std 

    M, m = data.max(axis=0), data.min(axis=0)

    normalized_data = (data - m) / (M - m)
    return standardized_data, normalized_data