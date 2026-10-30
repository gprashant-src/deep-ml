import numpy as np
def translate_object(points, tx, ty):
	points = np.array(points)

	points[:, 0] += tx
	points[:, 1] += ty
	return points
