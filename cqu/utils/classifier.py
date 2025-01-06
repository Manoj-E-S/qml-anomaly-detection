"""
Base 'BaseClassifier' class for all classifiers to inherit from. providing a common 
interface for all classifiers. This class is an abstract class and should not be
instantiated directly. 

Provided abstract methods:
    - fit(X, y): Fit the classifier to the given training data.
    - predict(X): Predict the class labels for the given data.
    - score(X, y): Return the accuracy of the classifier on the given data.
"""

from abc import ABC, abstractmethod
from typing import overload

import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import classification_report, roc_curve

from cqu.typing import Dataset

from .metrics import ClassifierMetrics


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


class BaseClassifier(ABC):
    random_state: int
    test_size: float

    def __init__(self, random_state: int = 42, test_size: float = 0.2) -> None:
        self.random_state = random_state
        self.test_size = test_size

    @overload
    def fit(self, X_train: Dataset, y_train: ArrayLike) -> None: ...

    @overload
    def fit(self, data: Dataset, target_column: str) -> None: ...

    @abstractmethod
    def fit(self, X_train_or_data: Dataset, y_train_or_target: ArrayLike | str) -> None:
        pass

    @overload
    def test(self, X_test: Dataset, y_test: ArrayLike) -> ClassifierMetrics: ...

    @overload
    def test(self, data: Dataset, target_column: str) -> ClassifierMetrics: ...

    @abstractmethod
    def test(
        self, X_test_or_data: Dataset, y_test_or_target: ArrayLike | str
    ) -> ClassifierMetrics:
        pass

    @abstractmethod
    def predict(self, X: Dataset) -> ArrayLike:
        pass

    def _handle_data_split(
        self, X_train_or_data: Dataset, y_train_or_target: ArrayLike | str
    ) -> tuple[pd.DataFrame, pd.Series]:
        if isinstance(y_train_or_target, str):
            X_train = X_train_or_data.drop(y_train_or_target, axis=1)
            y_train = X_train_or_data[y_train_or_target]
            return X_train, y_train
        else:
            return X_train_or_data, y_train_or_target
