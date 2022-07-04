import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from knntest import X_train  # , X_test, y_train, y_test


X, y = datasets.make_regression(
    n_samples=300, n_features=1, n_targets=1, noise=20
)  # , random_state=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)  # , shuffle=True)


# fig = plt.figure(figsize=(8, 6))  # (width, height)
# plt.scatter(X_train, y_train, color="red", label="train")  # , marker="o")
# plt.show()  # plt.scatter(X_test, y_test, color="blue", label="test")

from linear_regression import LinearRegression

regressor = LinearRegression(X_train, y_train)  # , X_test, y_test)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)  # , X_test)


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)  # , axis=0)


mse_value = mse(y_test, y_pred)  # , y_test)
print("MSE:", mse_value)
