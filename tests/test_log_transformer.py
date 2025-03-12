import pytest
import numpy as np
import pandas as pd
from src.feature_engineering import LogTransformer


def test_log_transformer_basic() -> None:
    """
    Test LogTransformer with positive numeric values.

    This test checks that the LogTransformer correctly applies the log1p
    transformation to each value in the DataFrame. It ensures that the
    result is numerically consistent with np.log1p applied to the same
    input values.

    Args:
        None

    Returns:
        None
    """
    X = pd.DataFrame({'Feature1': [0, 1, 10, 100]})
    transformer = LogTransformer()
    X_transformed = transformer.fit_transform(X)
    expected = np.log1p(X)
    np.testing.assert_array_almost_equal(X_transformed, expected.values, decimal=6)  # Allow slight floating-point error


def test_log_transformer_zeros() -> None:
    """
    Test LogTransformer to correctly handle zeros.

    This test ensures that zeros are handled correctly, as log1p(0)
    should be 0. The entire input DataFrame contains only zeros.

    Args:
        None

    Returns:
        None
    """
    X = pd.DataFrame({'Feature1': [0, 0, 0, 0]})
    transformer = LogTransformer()
    X_transformed = transformer.fit_transform(X)
    assert np.all(X_transformed == 0)  # log1p(0) = 0


def test_log_transformer_negative_values() -> None:
    """
    Test LogTransformer to raise an error when negative values are present.

    The LogTransformer should raise a ValueError when the input contains
    negative values since the logarithm is undefined for negative inputs.

    Args:
        None

    Returns:
        None
    """
    X = pd.DataFrame({'Feature1': [-1, -2, -3]})
    transformer = LogTransformer()
    with pytest.raises(ValueError, match="negative values"):
        transformer.fit_transform(X)


def test_log_transformer_non_numeric() -> None:
    """
    Test LogTransformer to raise an error for non-numeric data.

    This test ensures that a TypeError is raised when the input DataFrame
    contains non-numeric data such as strings.

    Args:
        None

    Returns:
        None
    """
    X = pd.DataFrame({'Feature1': ['a', 'b', 'c']})
    transformer = LogTransformer()
    with pytest.raises(TypeError):
        transformer.fit_transform(X)


def test_log_transformer_empty_dataframe() -> None:
    """
    Test LogTransformer with an empty DataFrame.

    This test checks that the LogTransformer raises a ValueError when
    attempting to transform an empty DataFrame.

    Args:
        None

    Returns:
        None
    """
    X = pd.DataFrame(columns=['Feature1'])
    transformer = LogTransformer()
    with pytest.raises(ValueError, match="at least one sample"):
        transformer.fit_transform(X)


def test_log_transformer_output_shape() -> None:
    """
    Test that LogTransformer output shape matches input shape.

    This test ensures that the transformed DataFrame maintains the
    same shape as the input DataFrame.

    Args:
        None

    Returns:
        None
    """
    X = pd.DataFrame({'Feature1': [1, 2, 3], 'Feature2': [10, 20, 30]})
    transformer = LogTransformer()
    X_transformed = transformer.fit_transform(X)
    assert X_transformed.shape == X.shape


def test_log_transformer_supports_numpy_array() -> None:
    """
    Test that LogTransformer works with NumPy arrays.

    This test verifies that the LogTransformer can process NumPy
    arrays as input and correctly applies the log1p transformation.

    Args:
        None

    Returns:
        None
    """
    X = np.array([[0, 1, 2], [3, 4, 5]])
    transformer = LogTransformer()
    X_transformed = transformer.fit_transform(X)
    expected = np.log1p(X)
    np.testing.assert_array_almost_equal(X_transformed, expected, decimal=6)  # Allow slight floating-point error
