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
    Identifies the most important features for multiple model types.

    Parameters:
    - data (pd.DataFrame): The input dataset.
    - target_column (str): The name of the target variable column.
    - top_n (int): Number of top features to return.

    Returns:
    - dict: A dictionary containing feature importance DataFrames for each model type.
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dictionary to store feature importances
    importance_results = {}

    # Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    importance = np.abs(model.coef_[0])
    feature_names = X.columns
    importance_results['logistic_regression'] = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False).head(top_n)

    # Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    importance = model.feature_importances_
    importance_results['random_forest'] = pd.DataFrame({'Feature': feature_names, 'Importance': importance})\
        .sort_values(by='Importance', ascending=False).head(top_n)

    # Gradient Boosting
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    importance = model.feature_importances_
    importance_results['gradient_boosting'] = pd.DataFrame({'Feature': feature_names, 'Importance': importance})\
        .sort_values(by='Importance', ascending=False).head(top_n)

    # Neural Network (using input layer weights)
    model = MLPClassifier(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    importance = np.mean(np.abs(model.coefs_[0]), axis=1)  # Use input layer weights
    importance_results['neural_network'] = pd.DataFrame({'Feature': feature_names, 'Importance': importance})\
        .sort_values(by='Importance', ascending=False).head(top_n)

    # KNN (using correlation as a proxy for feature importance)
    correlation = X.corrwith(y).abs()
    importance_results['knn'] = pd.DataFrame({'Feature': feature_names, 'Importance': correlation.values})\
        .sort_values(by='Importance', ascending=False).head(top_n)

    # Naive Bayes (using mean difference between class probabilities)
    model = GaussianNB()
    model.fit(X_train, y_train)
    importance = np.abs(model.theta_[0] - model.theta_[1])  # Mean difference between classes
    importance_results['naive_bayes'] = pd.DataFrame({'Feature': feature_names, 'Importance': importance})\
        .sort_values(by='Importance', ascending=False).head(top_n)

    # Print results
    for model_type, importance_df in importance_results.items():
        print(f"\n{model_type} Important Features:\n", importance_df)

    return importance_results

# Load your dataset
data = pd.read_csv("../datasets/ccfraud/creditcard.csv")
target_column = "Class"  # Replace with your target column name

# Get important features
important_features = get_feature_importance(data, target_column, top_n=10)
print(important_features)