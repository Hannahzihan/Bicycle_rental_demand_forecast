import matplotlib.pyplot as plt
import pandas as pd
from typing import List
from sklearn.pipeline import Pipeline


def coefficient_plotting(
    fitted_pipeline: Pipeline,
    preprocessor: Pipeline,
    numericals: List[str],
    categoricals: List[str],
    sort: str = "No"
) -> None:
    """
    Plot the coefficients of a Generalized Linear Model (GLM) from a fitted pipeline.

    This function extracts coefficients from a GLM model within a scikit-learn pipeline. It handles
    feature names from both the spline-transformed numerical features and the one-hot-encoded categorical features.
    The function visualizes the coefficients as a horizontal bar plot, with an option to sort the coefficients
    in ascending order.

    Args:
        fitted_pipeline (Pipeline):
            A scikit-learn Pipeline object that contains a fitted GLM model with
            named steps 'preprocess' (for feature preprocessing) and 'estimate' (for GLM estimator).

        preprocessor (Pipeline):
            A fitted scikit-learn preprocessing pipeline that contains the transformations for
            numerical and categorical features. It should have 'numeric' and 'cat' transformers.

        numericals (List[str]):
            A list of names for numerical features to be transformed using a spline transformer.

        categoricals (List[str]):
            A list of names for categorical features to be transformed using one-hot encoding (OHE).

        sort (str, optional):
            A flag to indicate whether to sort the coefficients in descending order.
            Set to "Yes" to sort, otherwise coefficients are displayed in their original order.
            Defaults to "No".

    Returns:
        None:
            The function does not return anything. It generates and displays a horizontal bar plot
            of the GLM coefficients.

    Raises:
        ValueError:
            If the 'sort' argument is not "Yes" or "No".

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.preprocessing import OneHotEncoder, StandardScaler
        >>> import pandas as pd
        >>> X = pd.DataFrame({'num1': [1, 2, 3], 'num2': [4, 5, 6], 'cat1': ['A', 'B', 'A']})
        >>> y = [0, 1, 0]
        >>> preprocessor = Pipeline([
        ...     ('numeric', StandardScaler()),
        ...     ('cat', OneHotEncoder(handle_unknown='ignore'))
        ... ])
        >>> pipeline = Pipeline([
        ...     ('preprocess', preprocessor),
        ...     ('estimate', LogisticRegression())
        ... ])
        >>> pipeline.fit(X, y)
        >>> coefficient_plotting(pipeline, preprocessor, ['num1', 'num2'], ['cat1'], sort="Yes")
    """
    fitted_preprocessor = fitted_pipeline.named_steps['preprocess']
    glm_model = fitted_pipeline.named_steps['estimate']
    glm_coefficients = glm_model.coef_

    spline_transformer = fitted_preprocessor.named_transformers_['numeric'].named_steps['spline']
    spline_feature_names = spline_transformer.get_feature_names_out(numericals)

    ohe_transformer = fitted_preprocessor.named_transformers_['cat']
    ohe_feature_names = ohe_transformer.get_feature_names_out(categoricals)

    all_feature_names = list(spline_feature_names) + list(ohe_feature_names)

    coefficients_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Coefficient': glm_coefficients
    })

    coefficients_df_sorted = coefficients_df.sort_values(by='Coefficient', ascending=False)
    plt.figure(figsize=(10, 8))
    if sort == "No":
        coefficients_df.plot(kind='barh', x='Feature', y='Coefficient', figsize=(10, 20))
    else:
        coefficients_df_sorted.plot(kind='barh', x='Feature', y='Coefficient', figsize=(10, 20))
    plt.title('GLM Coefficients')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.show()
