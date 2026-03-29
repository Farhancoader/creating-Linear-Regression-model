import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, epoches=5000):
        self.lr = lr
        self.epoches = epoches
        self.w = None
        self.b = None
        self.losses = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.epoches):
            y_pred = np.dot(X, self.w) + self.b
            error = y - y_pred

            dw = (-2/n_samples) * np.dot(X.T, error)
            db = (-2/n_samples) * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            loss = np.mean(error**2)
            self.losses.append(loss)

    def predict(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return np.dot(X, self.w) + self.b
