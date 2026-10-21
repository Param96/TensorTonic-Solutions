import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def train_logistic_regression(
    X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000
) -> tuple[np.ndarray, float]:
    """
    Trains logistic regression using gradient descent.

    Returns:
        w: Trained weights (n_features,)
        b: Trained bias (float)
    """
    n_samples, n_features = X.shape

    # Initialize weights and bias
    w = np.zeros(n_features)
    b = 0.0

    # Gradient descent
    for _ in range(steps):
        # Calculate predictions
        z = X @ w + b
        y_pred = _sigmoid(z)

        # Calculate gradients
        dw = (X.T @ (y_pred - y)) / n_samples
        db = np.mean(y_pred - y)

        # Update weights and bias
        w -= lr * dw
        b -= lr * db

    return w, float(b)
