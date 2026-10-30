def rejection_sampling_best_of_k(candidates, scores):
    """
    Select the highest-scoring candidate per prompt.

    Args:
        candidates: list of N lists, each containing K candidate outputs.
        scores: list of N lists, each containing K reward scores.

    Returns:
        List of N selected candidates.
    """
    ans = []
    for cand, score in zip(candidates, scores):
        arr = [(i, c, v) for i, (c, v) in enumerate(zip(cand, score))]

        res = max(arr, key=lambda x:(x[2], -x[0]))
        ans.append(res[1])

    return ans