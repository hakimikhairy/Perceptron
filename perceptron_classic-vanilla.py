import numpy as np

class Perceptron:
    def __init__(self, n_iterations=1000, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Trains model to learn input data X, based on X's labels of y.
        Parameters:
        X = Vector (np.ndarray)
        y = Vector (np.ndarray)
        """
        n_dimensions = X.shape[1]
        self.weights = np.zeros(n_dimensions)
        self.bias = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_combo = np.dot(x_i, self.weights) + self.bias
                y_prediction = np.sign(linear_combo)

                if y_prediction != y[idx]:
                    update = self.learning_rate * y[idx]
                    self.weights += update * x_i
                    self.bias += update

    def predict(self, X):
        """
        Classifies data of input X, based on trained perceptron model. Returns the classifications of input data X.
        Parameters:
        X = Vector (np.ndarray)
        """
        linear_combo = np.dot(X, self.weights) + self.bias

        return np.sign(linear_combo)