import numpy as np

def find_labels(X:np.ndarray, centroids:np.ndarray):
    sum_sq_X = np.sum(X ** 2, axis=1, keepdims=True)
    sum_sq_centroids = np.sum(centroids ** 2, axis=1)
    cross_term = np.dot(X, centroids.T)

    distances = sum_sq_X - 2 * cross_term + sum_sq_centroids
    labels = np.argmin(distances, axis=1)

    return labels


def segment_image(image, k: int, max_iters: int = 100):
    """
    Segment an RGB image using K-Means clustering on pixel colors.

    Args:
        image: nested list / array of shape (H, W, 3) with pixel values.
        k: number of clusters.
        max_iters: maximum number of K-Means iterations.

    Returns:
        Segmented image as a nested list of shape (H, W, 3).
    """
    image = np.array(image, dtype=float)

    arr = image.reshape(-1, 3)

    centroids = arr[:k].copy()
    pos = np.zeros(arr.shape[0])

    for _ in range(max_iters):
        labels = find_labels(arr, centroids)

        new_centroids = np.array([
            arr[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])

        centroids = new_centroids
    
    return centroids[labels].reshape(image.shape)


        

