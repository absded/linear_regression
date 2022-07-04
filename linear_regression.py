import numpy as np


class LinearRegression:
    def __init__(self, x, y):  # , x_test, y_test):
        self.x = x
        self.y = y
        self.m = 0
        self.b = 0
        self.n = len(x)
        self.x_mean = sum(x) / self.n  # np.mean(x)
        self.y_mean = sum(y) / self.n  # np.mean(y)
        self.x_y_mean = sum(
            [x[i] * y[i] for i in range(self.n)]
        )  # sum([x[i] * y[i] for i in range(self.n)])
        self.x_x_mean = sum(
            [x[i] * x[i] for i in range(self.n)]
        )  # sum([x[i] * x[i] for i in range(self.n)])
        self.x_y_mean_x_mean = sum(
            [x[i] * x[i] * y[i] for i in range(self.n)]
        )  # sum([x[i] * x[i] * y[i] for i in range(self.n)])
        self.x_y_mean_y_mean = sum(
            [x[i] * y[i] for i in range(self.n)]
        )  # sum([x[i] * y[i] for i in range(self.n)])
        self.x_y_mean_x_y_mean = sum(
            [x[i] * y[i] * y[i] for i in range(self.n)]
        )  # sum([x[i] * y[i] * y[i] for i in range(self.n)])
        self.x_y_mean_x_x_mean = sum(
            [x[i] * x[i] * y[i] for i in range(self.n)]
        )  # sum([x[i] * x[i] * y[i] for i in range(self.n)])
        self.x_y_mean_x_x_y_mean = sum(
            [x[i] * x[i] * y[i] * y[i] for i in range(self.n)]
        )  # sum([x[i] * x[i] * y[i] * y[i] for i in range(self.n)])
        self.x_y_mean_x_x_y_mean_x_x_mean = sum(
            [x[i] * x[i] * y[i] * y[i] * x[i] * x[i] for i in range(self.n)]
        )  # sum([x[i] * x[i] * y[i] * y[i] * x[i] * x[i] for i in range(self.n)])
        self.x_y_mean_x_x_y_mean_x_x_mean_x_x_mean = sum(
            [x[i] * x[i] * y[i] * y[i] * x[i] * x[i] * x[i] for i in range(self.n)]
        )  # sum([x[i] * x[i] * y[i] * y[i] * x[i] * x[i] * x[i] for i in range(self.n)])

    def fit(self, X, y):  # , x_test, y_test):
        n_sample, n_feature = X.shape
        self.weights = np.zeros(n_feature)
        self.bias = 0

        for _ in range(100):
            self.weights -= (
                self.gradient(X, y)
                / np.sqrt(
                    self.gradient_norm(X, y)
                )  # / np.sqrt(self.gradient_norm(X, y))
                + self.bias
                + self.weights
            )
            self.bias -= (
                self.gradient_bias(X, y)
                / np.sqrt(
                    self.gradient_norm(X, y)
                )  # / np.sqrt(self.gradient_norm(X, y))
                + self.bias
                + self.weights
            )

            dw = (1 / n_sample) * np.sum(self.gradient(X, y))
            db = (1 / n_sample) * np.sum(self.gradient_bias(X, y))

            self.weights -= dw / np.sqrt(self.gradient_norm(X, y))
            self.bias -= db / np.sqrt(self.gradient_norm(X, y))

    def predict(self, X):
        self.weights = np.array(self.weights) + self.bias
        return self.weights
