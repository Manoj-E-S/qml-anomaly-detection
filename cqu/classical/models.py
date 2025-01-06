from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from cqu.utils.metrics import ClassifierMetrics, get_metrics

from . import ClassicalModels


def _optimize_threshold(
    y_proba: ArrayLike, y_test: ArrayLike, step: float = 0.01, use_roc: bool = False
) -> float:
    """
    Find the optimal threshold that maximizes the F1-score for class 1.

    Parameters:
        y_proba (array): Predicted probabilities for the positive class.
        y_test (array): Ground truth labels.
        step (float): Increment for threshold adjustment.

    Returns:
        dict: Dictionary containing the optimal threshold, corresponding F1-score, and the classification report.
    """

    if use_roc:
        threshold = None
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        gmeans = (tpr * (1 - fpr)) ** 0.5
        optimal_idx = gmeans.argmax()
        return thresholds[optimal_idx]

    best_threshold = 0.0
    best_f1_score = 0.0

    # Iterate through thresholds from 0.0 to 1.0 with the given step size
    for threshold in [x * step for x in range(int(1 / step) + 1)]:
        y_pred = (y_proba >= threshold).astype(int)

        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        f1 = report["1"]["f1-score"]

        # Update best threshold if this F1 is better
        if f1 > best_f1_score:
            best_f1_score = f1
            best_threshold = threshold

    return best_threshold


def _test_train_split_helper(
    data: pd.DataFrame,
    feature_importances: Dict,
    target_column: str,
    random_state: int,
    test_size: int = 0.2,
):
    feature_names = feature_importances["feature"].tolist()

    X = data[feature_names]
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def logistic_regression_with_analysis(
    data: pd.DataFrame,
    target_column: str,
    feature_importances: Dict,
    random_state: int = 42,
    threshold: float | None = None,
) -> ClassifierMetrics:
    """
    Builds a logistic regression model using ROC curve analysis to determine the best threshold
    and class weights. Evaluates the model's performance on specified important features.

    Parameters:
    - data (pd.DataFrame): The input dataset.
    - target_column (str): The name of the target variable column.
    - important_features (pd.DataFrame): A DataFrame with feature names and their importance.
    - threshold (float): A predefined threshold. If None, optimal threshold will be determined using the ROC curve.

    Returns:
    - dict: A dictionary containing precision, recall, F1-score, ROC AUC score, and the classification report.
    """
    X_train, X_test, y_train, y_test = _test_train_split_helper(
        data, feature_importances, target_column, random_state
    )

    class_weights = {
        0: 1,
        1: 35,
    }

    model = LogisticRegression(
        max_iter=1000, random_state=random_state, class_weight=class_weights
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    if threshold is None:
        threshold = _optimize_threshold(y_proba, y_test, step=0.01, use_roc=True)

    y_pred = (y_proba >= threshold).astype(int)

    metrics = get_metrics(
        ClassicalModels.LOGISTIC_REGRESSION,
        y_test,
        y_pred,
        feature_importances,
        class_weights,
    )

    return metrics


def random_forest_with_analysis(
    data: pd.DataFrame,
    target_column: str,
    feature_importances: Dict,
    random_state: int = 42,
    threshold: float | None = None,
) -> ClassifierMetrics:
    """
    Builds a Random Forest model, optimizes the decision threshold using predict_proba, and
    computes the most optimal class weights based on class imbalance.

    Parameters:
    - data (pd.DataFrame): The input dataset.
    - target_column (str): The name of the target variable column.
    - important_features (pd.DataFrame): A DataFrame with feature names and their importance.
    - threshold (float): A predefined threshold. If None, optimal threshold will be determined using the ROC curve.

    Returns:
    - dict: A dictionary containing precision, recall, F1-score, ROC AUC score, and the classification report.
    """
    X_train, X_test, y_train, y_test = _test_train_split_helper(
        data, feature_importances, target_column, random_state
    )

    # class_weights = {0: len(y) / (2 * (y == 0).sum()), 1: len(y) / (2 * (y == 1).sum())}
    class_weights = {0: 1, 1: 35}

    model = RandomForestClassifier(
        n_estimators=100, random_state=random_state, class_weight=class_weights
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    if threshold is None:
        threshold = _optimize_threshold(y_proba, y_test, step=0.01, use_roc=True)

    y_pred = (y_proba >= threshold).astype(int)

    metrics = get_metrics(
        ClassicalModels.RANDOM_FOREST,
        y_test,
        y_pred,
        feature_importances,
        class_weights,
    )

    return metrics


def gradient_boosting_with_analysis(
    data: pd.DataFrame,
    target_column: str,
    feature_importances: Dict,
    random_state: int = 42,
    threshold: float | None = None,
) -> ClassifierMetrics:
    """
    Builds a Gradient Boosting model (XGBoost), optimizes the decision threshold using predict_proba,
    and computes the most optimal scale_pos_weight based on class imbalance.

    Parameters:
    - data (pd.DataFrame): The input dataset.
    - target_column (str): The name of the target variable column.
    - important_features (pd.DataFrame): A DataFrame with feature names and their importance.
    - threshold (float): A predefined threshold. If None, optimal threshold will be determined using the ROC curve.

    Returns:
    - dict: A dictionary containing precision, recall, F1-score, ROC AUC score, and the classification report.
    """
    X_train, X_test, y_train, y_test = _test_train_split_helper(
        data, feature_importances, target_column, random_state
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    if threshold is None:
        threshold = _optimize_threshold(y_proba, y_test, step=0.01, use_roc=True)

    y_pred = (y_proba >= threshold).astype(int)

    metrics = get_metrics(
        ClassicalModels.GRADIENT_BOOSTING, y_test, y_pred, feature_importances
    )

    return metrics


def neural_network_with_analysis(
    data: pd.DataFrame,
    target_column: str,
    feature_importances: Dict,
    random_state: int = 42,
    threshold: float | None = None,
) -> ClassifierMetrics:
    """
    Builds a Neural Network model, adjusts the class weights during training, optimizes the decision threshold,
    and computes evaluation metrics.

    Parameters:
    - data (pd.DataFrame): The input dataset.
    - target_column (str): The name of the target variable column.
    - important_features (pd.DataFrame): A DataFrame with feature names and their importance.
    - threshold (float): A predefined threshold. If None, optimal threshold will be determined using the ROC curve.

    Returns:
    - dict: A dictionary containing precision, recall, F1-score, ROC AUC score, and the classification report.
    """

    class FraudDetectionNN(nn.Module):
        def __init__(self, input_size):
            super(FraudDetectionNN, self).__init__()

            self.layers = nn.Sequential(
                nn.Linear(input_size, 64),  # Input Layer
                nn.ReLU(),  # Activation
                nn.BatchNorm1d(64),  # Batch Normalization
                nn.Dropout(0.3),  # Dropout Layer to prevent overfitting
                nn.Linear(64, 32),  # Hidden Layer 1
                nn.ReLU(),  # Activation
                nn.BatchNorm1d(32),  # Batch Normalization
                nn.Dropout(0.2),  # Dropout Layer
                nn.Linear(32, 16),  # Hidden Layer 2
                nn.ReLU(),  # Activation
                nn.BatchNorm1d(16),  # Batch Normalization
                nn.Linear(16, 1),  # Output Layer
                nn.Sigmoid(),  # Sigmoid activation for binary classification
            )

        def forward(self, x):
            return self.layers(x)

    X_train, X_test, y_train, y_test = _test_train_split_helper(
        data, feature_importances, target_column, random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(
        -1, 1
    )  # 1d tensor to 2d tensor
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)  # 1d tensor to 2d tensor

    class_counts = np.bincount(y_train)
    class_weights = {
        i: max(class_counts) / count for i, count in enumerate(class_counts)
    }

    model = FraudDetectionNN(X_train.shape[1])

    # Loss function and optimizer
    lossfn = nn.BCELoss()  # Binary Cross-Entropy Loss for classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train_tensor)
        loss = lossfn(outputs, y_train_tensor)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        threshold = _optimize_threshold(
            test_outputs.cpu().numpy().flatten(), y_test, 0.01
        )
        test_predictions_array = (
            test_outputs.cpu().numpy().flatten() >= threshold
        ).astype(
            int
        )  # Convert probabilities to binary predictions

        metrics = get_metrics(
            ClassicalModels.NEURAL_NETWORK,
            y_test,
            test_predictions_array,
            feature_importances,
            class_weights,
        )

    return metrics


def knn_model_with_analysis(
    data: pd.DataFrame,
    target_column: str,
    feature_importances: Dict,
    n_neighbors: int = 5,
    random_state: int = 42,
    threshold: float | None = None,
) -> ClassifierMetrics:
    """
    Builds a K-Nearest Neighbors (KNN) model, optimizes the threshold using predict_proba,
    and evaluates model performance.

    Parameters:
    - data (pd.DataFrame): The input dataset.
    - target_column (str): The name of the target variable column.
    - important_features (pd.DataFrame): A DataFrame with feature names and their importance.
    - n_neighbors (int): Number of neighbors to use for KNN.
    - threshold (float): A predefined threshold. If None, optimal threshold will be determined using ROC curve.

    Returns:
    - dict: A dictionary containing precision, recall, F1-score, ROC AUC score, and the classification report.
    """
    X_train, X_test, y_train, y_test = _test_train_split_helper(
        data, feature_importances, target_column, random_state
    )

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    knn.fit(X_train, y_train)

    y_proba = knn.predict_proba(X_test)[:, 1]

    threshold = _optimize_threshold(y_proba, y_test, 0.01)
    y_pred = (y_proba >= threshold).astype(int)

    metrics = get_metrics(ClassicalModels.KNN, y_test, y_pred, feature_importances)

    return metrics


def naive_bayes_model_with_analysis(
    data: pd.DataFrame,
    target_column: str,
    feature_importances: Dict,
    random_state: int = 42,
    threshold: float | None = None,
) -> ClassifierMetrics:
    """
    Builds a Naive Bayes model, optimizes the threshold using predict_proba,
    and evaluates model performance.

    Parameters:
    - data (pd.DataFrame): The input dataset.
    - target_column (str): The name of the target variable column.
    - important_features (pd.DataFrame): A DataFrame with feature names and their importance.
    - threshold (float): A predefined threshold. If None, optimal threshold will be determined using ROC curve.

    Returns:
    - dict: A dictionary containing precision, recall, F1-score, ROC AUC score, and the classification report.
    """
    X_train, X_test, y_train, y_test = _test_train_split_helper(
        data, feature_importances, target_column, random_state
    )

    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train, y_train)

    y_proba = naive_bayes.predict_proba(X_test)[:, 1]

    threshold = _optimize_threshold(y_proba, y_test, 0.01)
    y_pred = (y_proba >= threshold).astype(int)

    metrics = get_metrics(
        ClassicalModels.NAIVE_BAYES, y_test, y_pred, feature_importances
    )

    return metrics
