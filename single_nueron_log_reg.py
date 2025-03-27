

#Implement Single Nueron

import numpy as np

class Neuron:
    def __init__(self , learning_rate = 0.01, epochs = 100, batch_size = 5):
        self.learning_rate = learning_rate
        self.n_epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = None


    def fit(self, X,y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = np.zeros(1)

        n_iter = n_samples // self.batch_size #ideally divisible

        for _ in range(self.n_epochs):
            for j in range(n_iter):
                X_batch = X[j*self.batch_size:(j+1)*self.batch_size,:]
                Y_batch = y[j*self.batch_size:(j+1)*self.batch_size]

                linear_pred = np.dot(X_batch, self.weights) + self.bias
                y_predicted = self._sigmoid(linear_pred)
                dw = (1 / n_samples) * np.dot(X_batch.T, (y_predicted - Y_batch))
                db = (1 / n_samples) * np.sum(y_predicted - Y_batch)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

        print(f"{self.weights=}")
        print(f"{self.bias=}")

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_pred)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_pred]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


# Testing
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = Neuron()
    print(f"{X_train.shape=}")
    print(f"{X_test.shape=}")
    print(f"{y_train.shape=}")
    print(f"{y_test.shape=}")
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)
    print("LR classification accuracy:", accuracy(y_test, predictions))
