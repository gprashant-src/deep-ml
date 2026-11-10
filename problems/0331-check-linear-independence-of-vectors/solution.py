import numpy as np

def is_linearly_independent(vectors: list[list[float]]) -> bool:
    """
    Check if a set of vectors is linearly independent.
    
    Args:
        vectors: List of vectors, where each vector is a list of floats.
                 All vectors must have the same dimension.
        
    Returns:
        True if vectors are linearly independent, False otherwise.
    """
    if not vectors:return True
    # Your code here
    vectors = np.array(vectors, dtype=float)

    rank = np.linalg.matrix_rank(vectors)

    return rank == vectors.shape[0]