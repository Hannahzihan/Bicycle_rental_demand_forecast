import pandas as pd
from typing import Tuple

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load the training and testing datasets and return the training features, training labels, and testing features.

    Returns:
        X_train (pd.DataFrame): Training set features
        X_test (pd.DataFrame): Test set features
        y_train (pd.Series): Training set labels (churn prediction target variable)
    """
    # Read the datasets
    labeled_data = pd.read_csv(r"churn_prediction\data\train.csv")
    unlabeled_data = pd.read_csv(r"churn_prediction\data\test.csv")

    # Separate features and target variable
    X_train = labeled_data.drop(columns=["churned"])  # Features of the training dataset
    y_train = labeled_data["churned"]  # Target variable of the training dataset

    X_test = unlabeled_data  # Features of the test dataset

    return X_train, X_test, y_train
