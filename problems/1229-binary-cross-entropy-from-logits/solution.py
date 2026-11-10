import torch

def bce_with_logits(logits, targets):
    """Mean BCE-with-logits loss, numerically stable, rounded to 4 decimals.

    Args:
        logits (torch.Tensor): 1-D raw logits.
        targets (torch.Tensor): 1-D binary targets in {0, 1}, same shape.

    Returns:
        float: mean loss rounded to 4 decimal places.
    """
    # TODO: stable BCE-with-logits, mean, round to 4 decimals
    A = torch.where(logits > 0, logits, 0)
    B = logits * targets

    C = torch.log(1 + torch.exp(-torch.abs(logits)))

    ans = (A - B + C).mean().item()
    return round(ans, 4)
