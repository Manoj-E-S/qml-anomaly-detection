from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from cqu.typing import Dataset

from . import ClassicalModels


def logistic_regression_importance(
    X_train: Dataset, y_train: ArrayLike, extract_top_n: int = 5, random_state: int = 42
) -> pd.DataFrame:
    model = LogisticRegression(max_iter=5000, solver="saga", random_state=random_state)
    model.fit(X_train, y_train)
    importance = np.abs(model.coef_[0])
    feature_names = X_train.columns
    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values(by="importance", ascending=False)
        .head(extract_top_n)
    )


def random_forest_importance(
    X_train: Dataset, y_train: ArrayLike, extract_top_n: int = 5, random_state: int = 42
) -> pd.DataFrame:
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    importance = model.feature_importances_
    feature_names = X_train.columns
    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values(by="importance", ascending=False)
        .head(extract_top_n)
    )


def gradient_boosting_importance(
    X_train: Dataset, y_train: ArrayLike, extract_top_n: int = 5, random_state: int = 42
) -> pd.DataFrame:
    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    importance = model.feature_importances_
    feature_names = X_train.columns
    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values(by="importance", ascending=False)
        .head(extract_top_n)
    )


def neural_network_importance(
    X_train: Dataset, y_train: ArrayLike, extract_top_n: int = 5, random_state: int = 42
) -> pd.DataFrame:
    model = MLPClassifier(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    importance = np.mean(np.abs(model.coefs_[0]), axis=1)  # Input layer weights
    feature_names = X_train.columns
    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values(by="importance", ascending=False)
        .head(extract_top_n)
    )


def knn_importance(
    X_train: Dataset, y_train: ArrayLike, extract_top_n: int = 5, random_state: int = 42
) -> pd.DataFrame:
    # KNN has no inherent feature importance; use correlation as proxy
    # Random State is not used here
    correlation = X_train.corrwith(y_train).abs()
    importance = correlation.values
    feature_names = X_train.columns
    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values(by="importance", ascending=False)
        .head(extract_top_n)
    )


def naive_bayes_importance(
    X_train: Dataset, y_train: ArrayLike, extract_top_n: int = 5, random_state: int = 42
) -> pd.DataFrame:
    # Random State is not used here since GuassianNB is deterministic
    model = GaussianNB()
    model.fit(X_train, y_train)
    importance = np.abs(
        model.theta_[0] - model.theta_[1]
    )  # Mean difference between classes
    feature_names = X_train.columns
    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values(by="importance", ascending=False)
        .head(extract_top_n)
    )


model_to_function = {
    ClassicalModels.LOGISTIC_REGRESSION: logistic_regression_importance,
    ClassicalModels.RANDOM_FOREST: random_forest_importance,
    ClassicalModels.GRADIENT_BOOSTING: gradient_boosting_importance,
    ClassicalModels.NEURAL_NETWORK: neural_network_importance,
    ClassicalModels.KNN: knn_importance,
    ClassicalModels.NAIVE_BAYES: naive_bayes_importance,
}


def get_feature_importances(
    model_list: Dict[ClassicalModels, int],
    data: Dataset,
    target_column: str,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Identifies the most important features for a given model type.

    Parameters:
    - model_List (Dict[ClassicalModels, int]): A dictionary with model types as keys and the number of top features to return as values.
    - data (Dataset): A dataframe or array-like object containing the dataset.
    - target_column (str): The name of the target variable column.
    - random_state (int): The random seed to use for reproducibility.

    Returns:
    - Dict[str, pd.DataFrame]: A dict with model names as keys and the top features as values.
    """
    global model_to_function

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Train-test split
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    importance_results = {}

    # Execute functions in parallel
    with ThreadPoolExecutor() as executor:
        future_dict = {}

        for model_name, extract_top_n in model_list.items():
            future_dict[model_name] = executor.submit(
                model_to_function[model_name],
                X_train,
                y_train,
                extract_top_n,
                random_state,
            )

        for model_type, future in future_dict.items():
            importance_results[model_type] = future.result()

    return importance_results
