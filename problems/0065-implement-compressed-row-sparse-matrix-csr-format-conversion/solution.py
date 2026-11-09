import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

	:param dense_matrix: 2D list representing a dense matrix
	:return: A tuple containing (values array, column indices array, row pointer array)
	"""
	dense_matrix = np.array(dense_matrix)
	row, col = np.nonzero(dense_matrix)

	values = dense_matrix[row, col]

	row_counts = np.bincount(row, minlength=dense_matrix.shape[0])

	row_ptr = np.concatenate(([0], np.cumsum(row_counts)))

	return values.tolist(), col.tolist(), row_ptr.tolist()
