import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        # Initialize smoothing
        self.smoothing=smoothing

    def forward(self, X, y):
        # Fit model to binary features X and labels y
        N, D = X.shape
        # K = len(y)

        self.classes, self.counts = np.unique(y, return_counts=True)
        self.counts = np.log(self.counts / len(y))

        K = len(self.counts)

        self.model = np.zeros((K, D))

        for i, c in enumerate(self.classes):
            F = X[y == c]
            S = F.sum(axis=0)
            S_norm = (S + self.smoothing) / (len(F) + self.smoothing * 2)
            self.model[i] = np.log(S_norm)

    def predict(self, X):
        # Predict class labels for test set X
        log_p1 = self.model.copy()
        log_p0 = np.log1p(-np.exp(self.model))


        log_likelihood = (
            X[:, None, :] * log_p1[None, :, :]
            + (1 - X[:, None, :]) * log_p0[None, :, :]
        )

        S = log_likelihood.sum(axis=2)

        scores = S + self.counts[None, :]

        return self.classes[np.argmax(scores, axis=1)]
        