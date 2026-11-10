def matrix_determinant_and_trace(A: list[list[float]]) -> tuple[float, float]:
    """
    Compute the determinant and trace of a square matrix.
    
    Args:
        matrix: A square matrix (n x n) represented as list of lists
    
    Returns:
        Tuple of (determinant, trace)
    """
    # Your code here
    m = len(A)

    def det2(A:list):
        a, b, c, d = A[0][0], A[0][1], A[1][0], A[1][1]
        return a * d - b * c
    
    if m == 2:
        return det2(A), A[0][0] + A[1][1]
    
    DET = 0
    for j in range(m):
        minor = [
            [A[r][c] for c in range(m) if c != j]
            for r in range(1, m)
        ]

        sign = (-1) ** j
        DET += sign * A[0][j] * det2(minor)
    
    trace = sum(A[i][i] for i in range(m))
    return DET, trace