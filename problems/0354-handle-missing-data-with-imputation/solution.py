import numpy as np

def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Impute missing values in a 2D array using the specified strategy.
    
    Args:
        data: 2D numpy array with missing values represented as np.nan
        strategy: Imputation strategy - 'mean', 'median', or 'mode'
        
    Returns:
        2D numpy array with missing values imputed
    """
    # Your code here
    if strategy == "mean":
        col_means = np.nanmean(data, axis=0)
        idx = np.where(np.isnan(data))
        data[idx] = np.take(col_means, idx[1])
    
    elif strategy == "median":
        col_med = np.nanmedian(data, axis=0)
        idx = np.where(np.isnan(data))
        data[idx] = np.take(col_med, idx[1])
    
    elif strategy == "mode":
        for col_idx in range(data.shape[1]):
            column = data[:, col_idx]
        
            valid_values = column[~np.isnan(column)]
            
            if len(valid_values) > 0:
                vals, counts = np.unique(valid_values, return_counts=True)
                mode_val = vals[np.argmax(counts)]
                column[np.isnan(column)] = mode_val

    return data