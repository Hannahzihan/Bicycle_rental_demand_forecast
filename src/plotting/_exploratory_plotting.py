import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List, Optional


def plot_numerical_distributions(df: pd.DataFrame, numerical_features: List[str]) -> None:
    """
    Plot distribution charts for a list of numerical features using subplots.

    Args:
        df (pd.DataFrame): The DataFrame containing the numerical features.
        numerical_features (List[str]): A list of column names for the numerical features to be plotted.

    Returns:
        None: This function does not return anything. It displays the distribution plots.

    Example:
        >>> plot_numerical_distributions(df, ['Temperature', 'Humidity', 'Visibility'])
    """
    # Number of numerical features
    num_features = len(numerical_features)

    # Define the number of rows and columns for subplots
    cols = 3  # Number of columns in the subplot grid
    rows = (num_features // cols) + (num_features % cols > 0)  # Calculate rows needed

    # Set up the figure and axis objects
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

    # Plot each numerical feature
    for i, feature in enumerate(numerical_features):
        sns.histplot(df[feature], kde=True, bins=30, ax=axes[i], color='blue', edgecolor='black')
        axes[i].set_title(f"Distribution of {feature}", fontsize=12)
        axes[i].set_xlabel(feature, fontsize=10)
        axes[i].set_ylabel("Frequency", fontsize=10)

    # Remove any unused subplots
    for i in range(len(numerical_features), len(axes)):
        fig.delaxes(axes[i])

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_monthly_avg(df: pd.DataFrame) -> None:
    """
    Plot the average rented bike count by month using a bar chart.

    This function calculates the average rented bike count for each month and visualizes it
    as a bar plot. The months are ordered according to the standard calendar order.

    Args:
        df (pd.DataFrame): The DataFrame containing the following columns:
            - 'Month' (str): Month names as strings (e.g., 'January', 'February').
            - 'Rented Bike Count' (int): Count of rented bikes.

    Returns:
        None: This function does not return anything. It displays the bar chart.

    Example:
        >>> plot_monthly_avg(df)
    """
    # Group by Month and calculate the mean rented bike count
    monthly_avg = df.groupby('Month')['Rented Bike Count'].mean().reset_index()

    # Ensure the month order is correct according to the standard calendar order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']

    # Sort the dataframe according to month order
    monthly_avg['Month'] = pd.Categorical(monthly_avg['Month'], categories=month_order, ordered=True)
    monthly_avg = monthly_avg.sort_values('Month')

    # Set up the figure for the bar plot
    plt.figure(figsize=(10, 6))

    # Create a bar chart of the average rented bike count by month
    sns.barplot(data=monthly_avg, x='Month', y='Rented Bike Count', palette='viridis')

    # Add titles and axis labels
    plt.title('Average Rented Bike Count By Month')
    plt.xlabel('Month')
    plt.ylabel('Average Rented Bike Count')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Display the plot
    plt.show()


def plot_hourly_variation(df: pd.DataFrame, group_by: Optional[str] = "Week Status") -> None:
    """
    Plot the hourly variation of rented bike counts based on a specified grouping.

    This function visualizes the hourly variation of bike rentals using a line plot.
    The user can specify the grouping variable, such as 'Week Status', 'Day of Week',
    'Month', 'Seasons', or 'Holiday'.

    Args:
        df (pd.DataFrame): The DataFrame containing the following columns:
            - 'Hour' (int): Hour of the day (0-23).
            - 'Rented Bike Count' (int): Count of rented bikes.
            - 'Seasons', 'Month', 'Day of Week', 'Week Status', 'Holiday' (str):
              Categorical columns for grouping purposes.
        group_by (Optional[str]): The column name to group the data by.
            Valid options: 'Seasons', 'Month', 'Day of Week', 'Week Status', or 'Holiday'.
            Default is "Week Status".

    Raises:
        ValueError: If `group_by` is not one of the valid options.

    Returns:
        None: This function does not return anything. It displays the line plot.

    Example:
        >>> plot_hourly_variation(df, group_by='Seasons')
    """
    # Set up the figure size
    plt.figure(figsize=(12, 6))

    # Plot based on the specified grouping
    if group_by == "Seasons":
        sns.lineplot(data=df, x='Hour', y='Rented Bike Count', hue="Seasons", marker='o')
        plt.title('Hourly Bike Rentals Across Seasons')
    elif group_by == "Month":
        sns.lineplot(data=df, x='Hour', y='Rented Bike Count', hue="Month", marker='o')
        plt.title('Hourly Bike Rentals Across Month')
    elif group_by == "Day of Week":
        sns.lineplot(data=df, x='Hour', y='Rented Bike Count', hue="Day of Week", marker='o')
        plt.title('Hourly Bike Rentals Across Day of Week')
    elif group_by == "Week Status":
        sns.lineplot(data=df, x='Hour', y='Rented Bike Count', hue="Week Status", marker='o')
        plt.title('Hourly Bike Rentals Across Week Status')
    elif group_by == "Holiday":
        sns.lineplot(data=df, x='Hour', y='Rented Bike Count', hue="Holiday", marker='o')
        plt.title('Hourly Bike Rentals Across Holiday and No Holiday')
    else:
        raise ValueError(
            "Invalid group_by value. Choose from 'Seasons', 'Month', 'Day of Week', 'Week Status', or 'Holiday'.")

    # Add axis labels
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Rented Bike Count')

    # Display the plot
    plt.show()
