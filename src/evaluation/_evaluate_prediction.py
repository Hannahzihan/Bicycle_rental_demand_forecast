import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Dict
from sklearn.base import RegressorMixin


def evaluate_model(
    model: RegressorMixin,
    X: np.ndarray,
    y_true: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Evaluate a regression model on a given dataset and return predictions and performance metrics.

    This function takes a trained regression model, input features (X), and true target values (y_true)
    to compute predictions and evaluate model performance. It calculates key regression metrics
    (MAE, MSE, R2) and returns both the predictions and a dictionary of metrics.

    Args:
        model (RegressorMixin): The trained regression model to be evaluated.
        X (np.ndarray): The feature matrix (2D array) used for prediction.
        y_true (np.ndarray): The true target values corresponding to X.

    Returns:
        Tuple[np.ndarray, Dict[str, float]]:
            - **y_pred (np.ndarray)**: The predicted target values after inverse transformation.
            - **metrics (Dict[str, float])**: A dictionary containing evaluation metrics:
                - **'MAE' (float)**: Mean Absolute Error.
                - **'MSE' (float)**: Mean Squared Error.
                - **'R2' (float)**: R-squared score.

    Example:
        >>> from sklearn.linear_model import LinearRegression
        >>> import numpy as np
        >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        >>> y_true = np.array([2.5, 3.5, 4.5, 5.5])
        >>> model = LinearRegression().fit(X, np.log1p(y_true))  # Log transform for training
        >>> y_pred, metrics = evaluate_model(model, X, y_true)
        >>> print("Predictions:", y_pred)
        >>> print("Metrics:", metrics)
    """
    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)  # inverse of log1p
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return y_pred, {'MAE': mae, 'MSE': mse, 'R2': r2}
