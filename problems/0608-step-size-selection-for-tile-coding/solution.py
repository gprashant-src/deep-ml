import numpy as np

def tile_coding_step(base_alpha: float, n_tilings: int, weights: list, active_tiles: list, target: float) -> dict:
    """
    Perform a single value function update using tile coding with adjusted step size.
    
    Args:
        base_alpha: The base learning rate before adjustment
        n_tilings: Number of tilings used in the tile coding scheme
        weights: List of current weight values for all tiles
        active_tiles: List of indices of tiles active for the current state
        target: The target value (e.g., a return or TD target)
    
    Returns:
        Dictionary with 'adjusted_alpha', 'prediction_before',
        'prediction_after', and 'updated_weights'
    """
    
    weights = np.array(weights, dtype=float)
    step_size = base_alpha / n_tilings

    curr_v = weights[active_tiles]
    pred_old = curr_v.sum()

    weights[active_tiles] += step_size * (target - pred_old)

    pred_new = weights[active_tiles].sum()

    return {
        'adjusted_alpha': round(step_size, 4),
        'prediction_before': round(pred_old, 4),
        'prediction_after': round(pred_new, 4),
        'updated_weights': np.round(weights, 4).tolist()
    }