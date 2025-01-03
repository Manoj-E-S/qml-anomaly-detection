from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from cqu.classical_models.plotter import Plotter

def get_feature_importance(data, target_column, top_n=10):
    """
    Identifies the most important features for a given model type.

    Parameters:
    - data (pd.DataFrame): The input dataset.
    - target_column (str): The name of the target variable column.
    - model_type (str): Type of model (model_type, 'random_forest', 'gradient_boosting',
                        'neural_network', 'knn', 'naive_bayes').
    - top_n (int): Number of top features to return.

    Returns:
    - pd.DataFrame: A dataframe with the most important features and their importance scores.
    """

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    importance_results = {}

    # Define functions for each model
    def logistic_regression_importance():
        model = LogisticRegression(max_iter=5000, solver='saga', random_state=42)
        model.fit(X_train, y_train)
        importance = np.abs(model.coef_[0])
        feature_names = X.columns
        return 'logistic_regression', pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False).head(top_n)

    def random_forest_importance():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        importance = model.feature_importances_
        feature_names = X.columns
        return 'random_forest', pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False).head(top_n)

    def gradient_boosting_importance():
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)
        importance = model.feature_importances_
        feature_names = X.columns
        return 'gradient_boosting', pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False).head(top_n)

    def neural_network_importance():
        model = MLPClassifier(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        importance = np.mean(np.abs(model.coefs_[0]), axis=1)  # Input layer weights
        feature_names = X.columns
        return 'neural_network', pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False).head(top_n)

    def knn_importance():
        # KNN has no inherent feature importance; use correlation as proxy
        correlation = X.corrwith(y).abs()
        importance = correlation.values
        feature_names = X.columns
        return 'knn', pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False).head(top_n)

    def naive_bayes_importance():
        model = GaussianNB()
        model.fit(X_train, y_train)
        importance = np.abs(model.theta_[0] - model.theta_[1])  # Mean difference between classes
        feature_names = X.columns
        return 'naive_bayes', pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False).head(top_n)

    # Execute functions in parallel
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(logistic_regression_importance),
            executor.submit(random_forest_importance),
            executor.submit(gradient_boosting_importance),
            executor.submit(neural_network_importance),
            executor.submit(knn_importance),
            executor.submit(naive_bayes_importance),
        ]
        for future in futures:
            model_type, importance_df = future.result()
            importance_results[model_type] = importance_df

    return importance_results


if __name__ == '__main__':
    data = pd.read_csv("../datasets/ccfraud/creditcard.csv")
    target_column = "Class"

    print(get_feature_importance(data, target_column, top_n=5))


