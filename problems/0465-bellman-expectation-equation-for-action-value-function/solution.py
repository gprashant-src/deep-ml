import numpy as np

def bellman_q_value(states: list, actions: list, transition_probs: dict, rewards: dict, policy: dict, gamma: float, theta: float = 1e-6) -> dict:
    """
    Compute Q^pi(s, a) for all state-action pairs using iterative policy evaluation
    based on the Bellman expectation equation for action-values.

    Args:
        states: List of state identifiers
        actions: List of action identifiers
        transition_probs: Dict mapping (s, a, s') -> P(s'|s,a)
        rewards: Dict mapping (s, a, s') -> R(s,a,s')
        policy: Dict mapping (s, a) -> pi(a|s)
        gamma: Discount factor
        theta: Convergence threshold

    Returns:
        Dict mapping (s, a) -> Q-value rounded to 4 decimal places
    """
    Q = {(s, a): 0.0 for s in states for a in actions}

    while True:
        delta = 0.0
        Q_new = Q.copy()

        for s in states:
            for a in actions:
                Q_val = 0.0

                for s1 in states:
                    p = transition_probs.get((s, a, s1), 0.0)
                    if p == 0.0:continue

                    r = rewards.get((s, a, s1), 0.0)

                    E = 0.0
                    for a1 in actions:
                        pi = policy.get((s1, a1), 0.0)
                        if pi > 0:
                            E += pi * Q.get((s1, a1), 0.0)
                    
                    Q_val += p * (r + gamma * E)
                
                Q_new[(s, a)] = Q_val
                delta = max(delta, abs(Q_new[(s, a)] - Q[(s, a)]))
        Q = Q_new
        if delta < theta: break
    
    return {k: round(v, 4) for k, v in Q.items()}