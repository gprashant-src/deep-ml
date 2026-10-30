import numpy as np

def calculate_correlation_matrix(X, Y=None):
	# Your code here
	if Y is None:
		Y = X.copy()

	m  = X.shape[0]
	XS = (X - X.mean(axis=0)) / X.std(axis=0)
	YS = (Y - Y.mean(axis=0)) / Y.std(axis=0)

	return (XS.T @ YS) / m