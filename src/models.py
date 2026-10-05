import numpy as np



# HELPER

def _add_bias_column(X: np.ndarray) -> np.ndarray:
    """
    Prepend a column of ones to X.

    This lets a single weight vector w represent both the intercept and
    the slopes at once (the "homogeneous" representation): instead of
    predicting y = X @ w + b, we predict y = S @ w where
    S = [1 | X] and w's first entry plays the role of the intercept b.
    This way the closed-form formula w = (alpha*I + S^T S)^-1 S^T y
    directly gives us everything, with no separate bias term to track.
    """
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])



# RIDGE REGRESSION

class RidgeRegression:
    """Ridge regression, fit in closed form: w = (alpha*I + S^T S)^-1 S^T y."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha # regularization strength 
        self.w = None  # learned weights, set by fit()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        """Learn w by solving the ridge normal equations."""
        S = _add_bias_column(X)
        n_features_plus_bias = S.shape[1]

        # Normal equation of ridge: (alpha*I + S^T S) w = S^T y, where:
        #   A = alpha*I + S^T S   (the matrix that gets inverted)
        #   b = S^T y             
        A = self.alpha * np.eye(n_features_plus_bias) + S.T @ S
        b = S.T @ y

    
        try:
            self.w = np.linalg.solve(A, b) # if alpha > 0; instead of the inverse of A (to obtain w)
        except np.linalg.LinAlgError: # if alpha = 0 --> A not invertible --> infinite solutions
            self.w = np.linalg.pinv(A) @ b # solution: pseudo-inverse

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for new points: y_hat = [1 | X] @ w."""
        S = _add_bias_column(X) # add the column of ones (w includes the intercept)
        return S @ self.w

    # Mean squared error
    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean squared error of the current model on (X, y)."""
        y_pred = self.predict(X)
        return float(np.mean((y - y_pred) ** 2)) # float() converts a scalar



# LEAST SQUARES (special case of ridge)

class LeastSquaresRegression(RidgeRegression):
    """Ordinary (unregularized) least squares, i.e. the alpha=0 special case of ridge."""

    def __init__(self):
        super().__init__(alpha=0.0) # alpha = 0 --> no penalty term