import numpy as np

def scatter_add(values: np.ndarray, indices: np.ndarray, n_bins: int) -> np.ndarray:
    """Sum values into bins given by indices."""
    # Your code here
    idx = np.argsort(indices)
    
    val = values[idx]
    new_idx = indices[idx]

    result = np.zeros(n_bins)

    starts = np.r_[0, np.flatnonzero(np.diff(new_idx)) + 1]

    result[new_idx[starts]] = np.add.reduceat(
        val, starts
    )

    return result


