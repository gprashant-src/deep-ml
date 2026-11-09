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
        log_p1 = self.model
        log_p0 = np.log1p(-np.exp(log_p1))

        scores = X @ (log_p1 - log_p0).T
        scores += log_p0.sum(axis=1)
        scores += self.counts

        return self.classes[np.argmax(scores, axis=1)]
        