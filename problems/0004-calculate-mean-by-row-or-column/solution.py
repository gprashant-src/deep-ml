def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    m, n = len(matrix), len(matrix[0])
    if mode == "column":
        ans = [0] * n
        for j in range(n):
            S = sum(matrix[i][j] for i in range(m))
            ans[j] = S / m

    elif mode == "row":
        ans = [0] * m
        for i in range(m):
            S = sum(matrix[i][j] for j in range(n))
            ans[i] = S / n

    return ans

