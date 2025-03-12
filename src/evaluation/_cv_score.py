import numpy as np
from sklearn.metrics import mean_absolute_error
from typing import Union


def mae_inverse(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Calculate the Mean Absolute Error (MAE) for log-transformed predictions.

    This function first inverts the log-transformation on both `y_true` and `y_pred` using `np.expm1()`,
    and then computes the Mean Absolute Error (MAE) using the sklearn `mean_absolute_error` function.
    This is particularly useful for models where the target variable is log-transformed before fitting.

    Args:
        y_true (Union[np.ndarray, list]): The true target values in their log-transformed form.
        y_pred (Union[np.ndarray, list]): The predicted target values in their log-transformed form.

    Returns:
        float: The Mean Absolute Error (MAE) after inverting the log-transformation.

    Example:
        >>> import numpy as np
        >>> from sklearn.metrics import mean_absolute_error
        >>> y_true = np.log1p([1, 10, 100])
        >>> y_pred = np.log1p([2, 9, 105])
        >>> mae = mae_inverse(y_true, y_pred)
        >>> print(mae)
        2.3333333333333335
    """
    return mean_absolute_error(np.expm1(y_true), np.expm1(y_pred))
