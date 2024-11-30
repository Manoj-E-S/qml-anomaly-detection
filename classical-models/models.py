from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

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

    # Select and fit the model based on model_type
    model_type = 'logistic_regression'
    model = LogisticRegression(max_iter=5000, solver='saga', random_state=42)
    model.fit(X_train, y_train)
    importance = np.abs(model.coef_[0])
    feature_names = X.columns
    importance_results[model_type] = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False).head(top_n)

    model_type = 'random_forest'
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    importance = model.feature_importances_
    feature_names = X.columns
    importance_results[model_type] = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False).head(top_n)

    model_type = 'gradient_boosting'
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    importance = model.feature_importances_
    feature_names = X.columns
    importance_results[model_type] = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False).head(top_n)

    model_type = 'neural_network'
    model = MLPClassifier(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    importance = np.mean(np.abs(model.coefs_[0]), axis=1)  # Use input layer weights
    feature_names = X.columns
    importance_results[model_type] = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False).head(top_n)

    model_type = 'knn'
    # For KNN, no inherent feature importance; use correlation as proxy
    correlation = X.corrwith(y).abs()
    importance = correlation.values
    feature_names = X.columns
    importance_results[model_type] = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False).head(top_n)

    model_type = 'naive_bayes'
    # For Naive Bayes, use class-conditional probabilities as proxy for importance
    model = GaussianNB()
    model.fit(X_train, y_train)
    importance = np.abs(model.theta_[0] - model.theta_[1])  # Mean difference between classes
    feature_names = X.columns
    # Create a DataFrame with features and their importance scores
    importance_results[model_type] = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False).head(top_n)

    return importance_results


# Load your dataset
data = pd.read_csv("../datasets/ccfraud/creditcard.csv")
target_column = "Class"  # Replace with your target column name

print(get_feature_importance(data, target_column, top_n=5))
