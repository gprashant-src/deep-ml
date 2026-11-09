import numpy as np

def compressed_col_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

	:param dense_matrix: List of lists representing the dense matrix
	:return: Tuple of (values, row indices, column pointer)
	"""
	dense_matrix = np.array(dense_matrix, dtype=int)
	row, col = np.nonzero(dense_matrix)

	order = np.lexsort((row, col))

    values = dense_matrix[row[order], col[order]]
    row = row[order]

	col_counts = np.bincount(col, minlength=dense_matrix.shape[1])

	col_ptr = np.concatenate(([0], np.cumsum(col_counts)))

	return values.tolist(), row.tolist(), col_ptr.tolist()
