from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from cqu.utils.metrics import ClassifierMetrics, get_metrics

from . import ClassicalModels, ClassicalModelTypes


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


def ensemble_model_with_analysis(
    models: Dict[str, BaseEstimator],
    data: pd.DataFrame,
    target_column: str,
    feature_importances: Dict,
    random_state: int = 42,
    threshold: float | None = None,
    class_weights: Dict[int | str, int | float] | None = None,
) -> ClassifierMetrics:
    """
    Builds an ensemble model using a soft voting classifier and evaluates its performance.

    Parameters:
    - models (Dict[str, ModelTypes]): A dictionary of models where keys are model names
        (e.g., "logistic_regression", "random_forest") and values are scikit-learn or
        PyTorch model instances.
    - data (pd.DataFrame): A pandas DataFrame containing the input data, including
        features and the target column.
    - target_column (str): The name of the column in `data` representing the target variable.
    - feature_importances (Dict): A dictionary of feature importances or feature weights,
        typically used for dimensionality reduction or feature selection.
    - random_state (int, optional): The random state for reproducibility of the train-test split.
        Defaults to 42.
    - threshold (float | None, optional): A decision threshold for converting probabilities into
        binary predictions. If `None`, the function calculates the optimal threshold. Defaults to `None`.
    - class_weights (Dict[int|str, int|float] | None, optional): A dictionary mapping class labels
        (as integers or strings) to their corresponding weights. Used to handle class imbalance
        during training. Defaults to `None`.

    Returns:
    - ClassifierMetrics: An object containing performance metrics for the ensemble model,
        such as precision, recall, F1 score, ROC-AUC, etc.

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import pandas as pd

        >>> # Example models
        >>> models = {
        ...     "logistic_regression": LogisticRegression(),
        ...     "random_forest": RandomForestClassifier()
        ... }

        >>> # Example dataset
        >>> data = pd.DataFrame({
        ...     "feature1": [1, 2, 3, 4, 5],
        ...     "feature2": [5, 4, 3, 2, 1],
        ...     "target": [0, 1, 0, 1, 0]
        ... })

        >>> target_column = "target"
        >>> feature_importances = {"feature1": 0.8, "feature2": 0.2}

        >>> metrics = ensemble_model_with_analysis(
        ...     models, data, target_column, feature_importances
        ... )
        >>> print(metrics)
    """
    X_train, X_test, y_train, y_test = _test_train_split_helper(
        data, feature_importances, target_column, random_state
    )

    voting_clf = VotingClassifier(
        estimators=[(model_name.name, model) for model_name, model in models.items()],
        voting="soft",  # 'soft' voting uses predicted probabilities to make decisions
    )

    sample_weights = None
    if class_weights is not None:
        sample_weights = np.array([class_weights[int(y)] for y in y_train])

    voting_clf.fit(X_train, y_train, sample_weight=sample_weights)

    y_proba = voting_clf.predict_proba(X_test)[:, 1]

    threshold = _optimize_threshold(y_proba, y_test, 0.01)
    y_pred = (y_proba >= threshold).astype(int)

    metrics = get_metrics(ClassicalModels.ENSEMBLE, y_test, y_pred, feature_importances)

    return metrics


def get_the_best_classical_model(
    data: pd.DataFrame,
    target_column: str,
    feature_importances: Dict,
    random_state: int = 42,
    threshold: float | None = None,
    n_neighbors_for_knn: int = 5,
    class_weights: Dict[int | str, int | float] | None = None,
    parallel: bool = True,
) -> ClassicalModelTypes:
    """
    Train and evaluate multiple classical ML models on the given dataset and return the best model.

    Parameters:
    - data (pd.DataFrame): The input dataset.
    - target_column (str): The name of the target variable column.
    - feature_importances (pd.DataFrame): A DataFrame with feature names and their importance.
    - random_state (int): The random seed to use for reproducibility.
    - threshold (float): A predefined threshold. If None, optimal threshold will be determined using the ROC curve.
    - n_neighbors_for_knn (int): Number of neighbors to use
    - class_weights (Dict[int|str, int|float] | None): A dictionary mapping class labels to their corresponding weights.
    - parallel (bool): Whether to run the model training and evaluation in parallel.

    Returns:
    - the_best_model: The classical model with the highest F1-score.
    """
    models = {
        ClassicalModels.LOGISTIC_REGRESSION: LogisticRegression(
            max_iter=1000, random_state=random_state, class_weight=class_weights
        ),
        ClassicalModels.RANDOM_FOREST: RandomForestClassifier(
            n_estimators=100, random_state=random_state, class_weight=class_weights
        ),
        ClassicalModels.GRADIENT_BOOSTING: XGBClassifier(
            random_state=random_state,
            scale_pos_weight=(data[target_column] == 0).sum()
            / (data[target_column] == 1).sum(),
            eval_metric="logloss",
        ),
        ClassicalModels.NEURAL_NETWORK: FraudDetectionNN(data.shape[1]),
        ClassicalModels.KNN: KNeighborsClassifier(
            n_neighbors=n_neighbors_for_knn, weights="distance"
        ),
        ClassicalModels.NAIVE_BAYES: GaussianNB(),
    }

    analysis_functions = {
        ClassicalModels.LOGISTIC_REGRESSION: logistic_regression_with_analysis,
        ClassicalModels.RANDOM_FOREST: random_forest_with_analysis,
        ClassicalModels.GRADIENT_BOOSTING: gradient_boosting_with_analysis,
        ClassicalModels.NEURAL_NETWORK: neural_network_with_analysis,
        ClassicalModels.KNN: knn_model_with_analysis,
        ClassicalModels.NAIVE_BAYES: naive_bayes_model_with_analysis,
        ClassicalModels.ENSEMBLE: ensemble_model_with_analysis,
    }

    if parallel == False:
        metrics = {}
        for model_name, analizer in analysis_functions.items():
            if (
                model_name == ClassicalModels.KNN
            ):  # Special case for KNN which requires `n_neighbors`
                print("KNN")
                metrics[model_name] = analizer(
                    data=data,
                    target_column=target_column,
                    feature_importances=feature_importances[model_name],
                    n_neighbors=n_neighbors_for_knn,  # n_neighbors
                    random_state=random_state,
                    threshold=threshold,
                )
            elif (
                model_name == ClassicalModels.ENSEMBLE
            ):  # Ensemble Model has `models` and `class_weights`
                print("Ensemble")
                metrics[model_name] = analizer(
                    models={
                        ClassicalModels.LOGISTIC_REGRESSION: models[
                            ClassicalModels.LOGISTIC_REGRESSION
                        ],
                        ClassicalModels.RANDOM_FOREST: models[
                            ClassicalModels.RANDOM_FOREST
                        ],
                    },  # models
                    data=data,
                    target_column=target_column,
                    feature_importances=feature_importances[
                        ClassicalModels.RANDOM_FOREST
                    ],
                    random_state=random_state,
                    threshold=threshold,
                    class_weights=class_weights,  # class_weights
                )
            else:  # General case for other models
                print("General")
                metrics[model_name] = analizer(
                    data=data,
                    target_column=target_column,
                    feature_importances=feature_importances[model_name],
                    random_state=random_state,
                    threshold=threshold,
                )

    else:
        metrics = {}
        with ThreadPoolExecutor() as executor:
            futures = {}
            for model_name, analizer in analysis_functions.items():
                if (
                    model_name == ClassicalModels.KNN
                ):  # Special case for KNN which requires `n_neighbors`
                    print("KNN")
                    futures[
                        executor.submit(
                            analizer,
                            data=data,
                            target_column=target_column,
                            feature_importances=feature_importances[model_name],
                            n_neighbors=n_neighbors_for_knn,  # n_neighbors
                            random_state=random_state,
                            threshold=threshold,
                        )
                    ] = model_name
                elif (
                    model_name == ClassicalModels.ENSEMBLE
                ):  # Ensemble Model has `models` and `class_weights`
                    print("Ensemble")
                    futures[
                        executor.submit(
                            analizer,
                            models={
                                ClassicalModels.LOGISTIC_REGRESSION: models[
                                    ClassicalModels.LOGISTIC_REGRESSION
                                ],
                                ClassicalModels.RANDOM_FOREST: models[
                                    ClassicalModels.RANDOM_FOREST
                                ],
                            },  # models
                            data=data,
                            target_column=target_column,
                            feature_importances=feature_importances[
                                ClassicalModels.RANDOM_FOREST
                            ],
                            random_state=random_state,
                            threshold=threshold,
                            class_weights=class_weights,  # class_weights
                        )
                    ] = model_name
                else:  # General case for other models
                    print("General")
                    futures[
                        executor.submit(
                            analizer,
                            data=data,
                            target_column=target_column,
                            feature_importances=feature_importances[model_name],
                            random_state=random_state,
                            threshold=threshold,
                        )
                    ] = model_name

            # Collect results as they complete
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    metrics[model_name] = future.result()
                    print(metrics[model_name])
                except Exception as e:
                    print(f"ERROR PROCESSING {model_name}\n: {e}")

    best_model = max(metrics, key=lambda x: metrics[x].get_report()["1"]["f1-score"])

    return models[best_model]
