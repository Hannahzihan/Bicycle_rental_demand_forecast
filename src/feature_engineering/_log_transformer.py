from typing import Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to apply log transformation to the selected features.
    Uses log1p(x) = log(x + 1) to handle zero values.
    """

    def __init__(self) -> None:
        """Initialize the LogTransformer. No parameters are required."""
        pass

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Optional[Union[pd.Series, np.ndarray]] = None) -> 'LogTransformer':
        """
        No fitting required for this transformer.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            The input data to fit. It is not used in this method but is required to comply with
            sklearn's BaseEstimator and TransformerMixin API.

        y : Optional[Union[pd.Series, np.ndarray]], optional
            The target variable. Not used in this method.

        Returns
        -------
        self : LogTransformer
            Returns the instance itself.
        """
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Apply log1p transformation to the input features.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            The input data to transform. Must be non-negative.

        Returns
        -------
        np.ndarray
            The log-transformed data as a NumPy array.

        Raises
        ------
        ValueError
            If the input array is empty.

        ValueError
            If the input array contains negative values, as log1p cannot process negative numbers.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values  # Convert DataFrame to NumPy array
        if X.size == 0:
            raise ValueError("Input DataFrame must have at least one sample.")
        if np.any(X < 0):
            raise ValueError("Input contains negative values, which cannot be transformed using log1p.")

        return np.log1p(X)
