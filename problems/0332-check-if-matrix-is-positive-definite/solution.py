import numpy as np

def check_positive_definite(matrix: list) -> dict:
    """
    Check if a matrix is positive definite and compute its eigenvalues.
    
    Args:
        matrix: A 2D list representing a square matrix
        
    Returns:
        dict with 'is_positive_definite' (bool) and 'eigenvalues' (list of floats sorted ascending)
    """
    # Your code here
    A = np.array(matrix, dtype=float)
    E = np.linalg.eigvals(A).real

    return {
        "is_positive_definite": bool(np.allclose(A, A.T) and np.all(E > 0)), 
        "eigenvalues": list(np.sort(np.round(E, 4)))
    }
