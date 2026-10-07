import numpy as np



# HELPER

def _add_bias_column(X: np.ndarray) -> np.ndarray: # returns a matrix; _ : internal function
    """
    Prepend a column of ones to X.

    This lets a single weight vector w represent both the intercept and
    the slopes at once (the "homogeneous" representation): instead of
    predicting y = X @ w + b, we predict y = S @ w where
    S = [1 | X] and w's first entry plays the role of the intercept b.
    This way the closed-form formula w = (alpha*I + S^T S)^-1 S^T y
    directly gives us everything, with no separate bias term to track.
    """
    ones = np.ones((X.shape[0], 1)) # column of 1 with n. of rows of X
    return np.hstack([ones, X]) # h = horizontal: X = n × d --> n × (d+1).



# RIDGE REGRESSION

class RidgeRegression:
    """Ridge regression, fit in closed form: w = (alpha*I + S^T S)^-1 S^T y."""

    def __init__(self, alpha: float = 1.0): # constructor
        self.alpha = alpha # regularization strength 
        self.w = None  # weights not yet --> set by fit()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        S = _add_bias_column(X)
        n_features_plus_bias = S.shape[1] # n. of cols

        # Normal equation of ridge: (alpha*I + S^T S) w = S^T y, where:
        #   A = alpha*I + S^T S   (the matrix that gets inverted)
        #   b = S^T y 
        A = self.alpha * np.eye(n_features_plus_bias) + S.T @ S
        b = S.T @ y

        try:
            self.w = np.linalg.solve(A, b) # if alpha > 0 --> A always invertible
        except np.linalg.LinAlgError: # if alpha = 0 --> A may be not invertible
            self.w = np.linalg.pinv(A) @ b # solution: pseudo-inverse (eg. minmum norm)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray: 
        S = _add_bias_column(X) # add the column of ones
        return S @ self.w # vector with predictions for each row of S

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return float(np.mean((y - y_pred) ** 2))



# LEAST SQUARES (special case of ridge)

class LeastSquaresRegression(RidgeRegression):
    """Ordinary (unregularized) least squares, i.e. the alpha=0 special case of ridge."""

    def __init__(self):
        super().__init__(alpha=0.0) # fixes alpha = 0 --> no penalty term