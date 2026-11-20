import numpy as np

def ipo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):
    """
    Identity Preference Optimization (IPO) loss.

    Args:
        policy_chosen_logps: log πθ(y_w) per example
        policy_rejected_logps: log πθ(y_l) per example
        ref_chosen_logps: log πref(y_w) per example
        ref_rejected_logps: log πref(y_l) per example
        beta: KL temperature (target gap is 1/(2β))

    Returns:
        Mean IPO loss, rounded to 6 decimals.
    """
    # Your code here
    H = [
        (theta_w - ref_w) - (theta_l - ref_l) for theta_w, ref_w, theta_l, ref_l in zip(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps)
    ]
    target_gap = 1 / (2 * beta)
    
    n = len(H)

    L = [(h - target_gap) ** 2 for h in H]

    return round(sum(L) / n, 6)
