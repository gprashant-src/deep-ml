import numpy as np

def bucketize(v: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Return bucket indices for v given sorted boundaries."""
    # Your code here
    
    mask = v[:, None] >= boundaries
    return np.sum(mask, axis=1)



    # low_mask = v < boundaries[0]
    # high_mask = v >= boundaries[-1]

    # ans = v.copy()

    # ans[low_mask] = 0
    # ans[high_mask] = len(boundaries)

    # for k in range(len(boundaries)):
    #     mid_mask = (boundaries[k - 1] <= v) & (v < boundaries[k])
    #     ans[mid_mask] = k
    
    # return ans.astype(int)
    

