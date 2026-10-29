import numpy as np

def detect_outliers_iqr(data: list[float], k: float = 1.5) -> dict:
	"""
	Detect and remove outliers using the IQR method.
	
	Args:
		data: List of numerical values
		k: IQR multiplier for determining outlier bounds (default 1.5)
	
	Returns:
		Dictionary with 'cleaned_data', 'outlier_indices', 'lower_bound', 'upper_bound'
	"""
	# Your code here
	data = np.array(data)
	# med = np.median(data)

	# q1
	# lower = data[data < med]
	q1 = np.percentile(data, 25)

	# q3
	# upper = data[data > med]
	q3 = np.percentile(data, 75)

	iqr = q3 - q1

	lb = q1 - k * iqr
	ub = q3 + k * iqr

	outliers = np.where((data < lb) | (data > ub))

	return {
		"cleaned_data": data[(data >= lb) & (data <= ub)].tolist(),
		"outlier_indices": list(outliers[0]),
		"lower_bound": float(lb),
		"upper_bound": float(ub)
	}




