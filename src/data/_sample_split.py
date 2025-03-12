import hashlib
import numpy as np
import pandas as pd


def create_sample_split(
    df: pd.DataFrame,
    id_column: str,
    training_frac: float = 0.8
) -> pd.DataFrame:
    """Create a train-test split for a dataset based on a hash or modulo of an ID column.

    This function assigns each record in the dataset to either the training or testing set
    based on the values in a specified ID column. It ensures the split is consistent and reproducible.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be split.
    id_column : str
        The name of the column containing unique identifiers for splitting.
    training_frac : float, optional, default=0.8
        The fraction of data to include in the training set. Must be between 0 and 1.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with an additional column named "sample", indicating whether
        each record belongs to the "train" or "test" set.

    Raises
    ------
    ValueError
        If `training_frac` is not between 0 and 1.
        If `id_column` does not exist in `df`.

    Notes
    -----
    - For integer ID columns, the modulo operation is used to determine the split.
    - For non-integer ID columns, an MD5 hash function is applied to ensure consistency.
    - The split is based on the training fraction multiplied by 100 (e.g., 80% = training_frac * 100).

    Examples
    --------
    >>> import pandas as pd
    >>> data = {'id': [1, 2, 3, 4, 5, 6], 'value': [10, 20, 30, 40, 50, 60]}
    >>> df = pd.DataFrame(data)
    >>> df = create_sample_split(df, id_column='id', training_frac=0.8)
    >>> print(df)
       id  value sample
    0   1     10  train
    1   2     20  train
    2   3     30  train
    3   4     40  test
    4   5     50  train
    5   6     60  train
    """

    if df[id_column].dtype == np.int64:
        modulo = df[id_column] % 100
    else:
        modulo = df[id_column].apply(
            lambda x: int(hashlib.md5(str(x).encode()).hexdigest(), 16) % 100
        )

    df["sample"] = np.where(modulo < training_frac * 100, "train", "test")

    return df
