import numpy as np

class LinearRegression:
    def __init__(self, learning_rate = 0.001, n_iters = 1000):
        self.l_r = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X,y ):
        n_samples , n_features = X.shape
        print(f"{X.shape=}")
        self.weights = np.zeros(n_features)
        self.bias = np.zeros(1)
        print(f"{self.weights=}")
        print(f"{self.bias=}")
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.matmul(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.l_r * dw
            self.bias -= self.l_r * db
        print(f"{self.weights=}")
        print(f"{self.bias=}")

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

    def r2_score(self, y_true, y_pred):
        corr_matrix = np.corrcoef(y_true, y_pred)
        return corr_matrix[0, 1]


#TEST CODE
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets


    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


    X, y = datasets.make_regression(
        n_samples=100, n_features=2, noise=20, random_state=4
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = LinearRegression(learning_rate=0.01, n_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    print("MSE:", mse)

    accu = regressor.r2_score(y_test, predictions)
    print("Accuracy:", accu)