import numpy as np

def random_sample_greedy_search(q_table: np.ndarray, states: list, sample_size: int, epsilon: float, seed: int) -> dict:
    """
    Perform Random Sample Greedy action selection for a batch of states.
    
    Args:
        q_table: np.ndarray of shape (num_states, num_actions), Q-value estimates
        states: list of int, state indices to select actions for
        sample_size: int, number of actions to sample for greedy selection
        epsilon: float, probability of selecting a uniformly random action
        seed: int, random seed for reproducibility
    
    Returns:
        dict with keys:
            'actions': list of int, selected action indices
            'q_values': list of float, Q-values for selected actions (rounded to 4 dp)
            'greedy_match_rate': float, fraction matching true greedy (rounded to 4 dp)
    """
    rng = np.random.default_rng(seed)

    n = q_table.shape[1]
    actions, q_values, matches = [], [], []

    for s in states:
        if rng.random() < epsilon:
            a = rng.integers(n)
        else:
            idx = np.sort(rng.choice(n, size=sample_size, replace=False))
            q = q_table[s, idx]
            a = idx[np.argmax(q)]

        actions.append(int(a))
        q_values.append(round(float(q_table[s, a]), 4))

        mx = np.max(q_table[s])
        matches.append(q_table[s, a] == mx)

    return {
        "actions": actions,
        "q_values": q_values,
        "greedy_match_rate": round(float(np.mean(matches)), 4)
    }