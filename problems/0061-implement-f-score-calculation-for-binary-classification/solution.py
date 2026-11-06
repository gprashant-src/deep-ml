import numpy as np

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	actual = y_true.sum()
	predicted = y_pred.sum()

	true = ((y_true == 1) & (y_pred == 1)).sum()

	P, R = true / predicted, true / actual

	F = (1 + beta ** 2) * P * R / ((beta ** 2) * P + R)
	return round(F, 3)

