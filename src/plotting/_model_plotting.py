import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union


def plot_predicted_vs_actual(y_true: Union[np.ndarray, list],
                             y_pred: Union[np.ndarray, list],
                             title: str) -> None:
    """
    Plot a scatter plot comparing actual vs. predicted values for model evaluation.

    This plot visualizes how well the predicted values align with the actual values.
    A red dashed line representing the line of perfect equality (y = x) is included
    to show ideal predictions.

    Args:
        y_true (Union[np.ndarray, list]): The true target values.
        y_pred (Union[np.ndarray, list]): The predicted values from the model.
        title (str): Title of the plot.

    Returns:
        None: This function only produces a plot and does not return any value.

    Example:
        >>> y_true = [10, 20, 30, 40]
        >>> y_pred = [12, 18, 29, 42]
        >>> plot_predicted_vs_actual(y_true, y_pred, 'Actual vs Predicted')
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()


def plot_multi_model_lorenz_curve(y_true: Union[np.ndarray, list],
                                  model_predictions: List[Union[np.ndarray, list]],
                                  model_labels: List[str],
                                  title: str = 'Lorenz Curve for Model Predictions') -> None:
    """
    Plot the Lorenz curves for multiple models and include Gini index in the legend.

    The Lorenz curve visualizes the cumulative distribution of model predictions
    relative to the actual outcomes. The Gini index, a measure of inequality,
    is displayed in the legend for each model. A random baseline is included as
    a dashed line for reference.

    Args:
        y_true (Union[np.ndarray, list]): The true target values.
        model_predictions (List[Union[np.ndarray, list]]): A list of arrays containing predictions from multiple models.
        model_labels (List[str]): A list of model names to be used in the legend.
        title (str, optional): Title of the plot. Defaults to 'Lorenz Curve for Model Predictions'.

    Returns:
        None: This function only produces a plot and does not return any value.

    Example:
        >>> y_true = [10, 20, 30, 40]
        >>> model_predictions = [[8, 22, 28, 35], [12, 18, 31, 39]]
        >>> model_labels = ['Model A', 'Model B']
        >>> plot_multi_model_lorenz_curve(y_true, model_predictions, model_labels)
    """
    y_true = np.array(y_true)  # Ensure y_true is a numpy array

    # Create the line of perfect equality
    perfect_line = np.linspace(0, 1, len(y_true))
    plt.figure(figsize=(10, 6))
    plt.plot(perfect_line, perfect_line, 'k--', label='Random baseline')

    for predictions, label in zip(model_predictions, model_labels):
        predictions = np.array(predictions)  # Ensure predictions are numpy arrays

        # Sort indices by prediction values
        sorted_indices = np.argsort(predictions)

        sorted_pred = predictions[sorted_indices]

        # Calculate cumulative sums
        cum_pred = np.cumsum(sorted_pred) / sum(sorted_pred)

        # Calculate the Gini index
        area_under_lorenz = np.trapz(cum_pred, perfect_line)
        area_under_line = 0.5  # Area under the line of perfect equality
        gini_index = 2 * (area_under_line - area_under_lorenz)

        # Plotting
        plt.plot(perfect_line, cum_pred, label=f'{label} (Gini index: {gini_index:.3f})')

    plt.title(title)
    plt.xlabel('Fraction of observations (ordered by model from lowest to highest outcome)')
    plt.ylabel('Fraction of total outcome')
    plt.legend(title="Models")
    plt.grid(True)
    plt.show()
