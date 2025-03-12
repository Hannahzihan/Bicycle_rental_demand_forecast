import pytest
import numpy as np
import pandas as pd
from src.feature_engineering import Discretizer, SparseFeatureCombiner


@pytest.mark.parametrize("n_bins, strategy", [(3, 'quantile'), (5, 'uniform')])
def test_discretizer_bins(n_bins: int, strategy: str) -> None:
    """
    Test Discretizer to correctly bin continuous values into a specified number of bins.

    Args:
        n_bins (int): Number of bins to split the continuous values into.
        strategy (str): Strategy to determine the bin edges. Can be 'uniform', 'quantile', or 'kmeans'.

    Asserts:
        - The shape of the transformed DataFrame is the same as the input DataFrame.
        - All transformed values are non-negative.
        - All transformed values are less than the number of bins.
    """
    X = pd.DataFrame({'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    discretizer = Discretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    X_transformed = discretizer.fit_transform(X)
    assert X_transformed.shape == X.shape  # Shape should remain the same
    assert np.all(X_transformed >= 0)  # Bins should be non-negative
    assert np.all(X_transformed < n_bins)  # Bins should be less than n_bins


def test_discretizer_zero_variance() -> None:
    """
    Test Discretizer with a zero-variance (constant) column.

    Asserts:
        - The shape of the transformed DataFrame is the same as the input DataFrame.
        - All rows are placed in the same bin (all zeros) because all values are identical.
    """
    X = pd.DataFrame({'Feature1': [5, 5, 5, 5, 5]})
    discretizer = Discretizer(n_bins=3, encode='ordinal', strategy='uniform')
    X_transformed = discretizer.fit_transform(X)
    assert X_transformed.shape == X.shape
    assert np.all(X_transformed == 0)  # All rows will belong to the same bin


def test_sparse_feature_combiner() -> None:
    """
    Test SparseFeatureCombiner to combine Rainfall and Snowfall into binary values.

    The combined feature will be 1 if either 'Rainfall' or 'Snowfall' has a non-zero value.

    Asserts:
        - The combined feature correctly identifies when rain or snow occurs (value of 1)
          or when neither occurs (value of 0).
    """
    X = pd.DataFrame({'Rainfall': [0, 2, 0, 0, 3.5], 'Snowfall': [0, 0, 0.5, 0, 0]})
    combiner = SparseFeatureCombiner()
    combined_feature = combiner.fit_transform(X)
    expected_output = np.array([[0], [1], [1], [0], [1]])
    np.testing.assert_array_equal(combined_feature, expected_output)


def test_discretizer_empty_dataframe() -> None:
    """
    Test Discretizer with an empty DataFrame.

    Asserts:
        - A ValueError is raised when an empty DataFrame is passed to the Discretizer.
    """
    X = pd.DataFrame(columns=['Feature1'])
    discretizer = Discretizer(n_bins=3, encode='ordinal', strategy='uniform')
    with pytest.raises(ValueError):
        discretizer.fit_transform(X)


def test_sparse_feature_combiner_empty_dataframe() -> None:
    """
    Test SparseFeatureCombiner with an empty DataFrame.

    Asserts:
        - A ValueError is raised when an empty DataFrame is passed to the SparseFeatureCombiner.
    """
    X = pd.DataFrame(columns=['Rainfall', 'Snowfall'])
    combiner = SparseFeatureCombiner()
    with pytest.raises(ValueError):
        combiner.fit_transform(X)
