from collections import Counter

def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    H = Counter(samples)

    N = len(samples)

    return sorted([
        (k, v / N) for k, v in H.items()
    ], key=lambda x: x[0])
    