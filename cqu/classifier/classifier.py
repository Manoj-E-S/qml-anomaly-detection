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
from typing import Any, TypeAlias, overload

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

Dataset: TypeAlias = ArrayLike | pd.DataFrame


class BaseClassifier(ABC):
    @overload
    def fit(self, X_train: Dataset, y_train: ArrayLike) -> None: ...

    @overload
    def fit(self, data: Dataset, target_column: str) -> None: ...

    @abstractmethod
    def fit(self, X_train_or_data: Dataset, y_train_or_target: ArrayLike | str) -> None:
        pass

    @overload
    def test(self, X_test: Dataset, y_test: ArrayLike) -> Any: ...

    @overload
    def test(self, data: Dataset, target_column: str) -> Any: ...

    @abstractmethod
    def test(self, X_train_or_data: Dataset, y_test_or_target: ArrayLike | str) -> Any:
        pass

    @abstractmethod
    def predict(self, X: Dataset) -> ArrayLike:
        pass

    def __handle_data_split(
        self, X_train_or_data: Dataset, y_test_or_target: ArrayLike | str
    ) -> tuple[pd.DataFrame, pd.Series]:
        if isinstance(y, str):
            X = X_train_or_data.drop(y_test_or_target, axis=1)
            y = X_train_or_data[y_test_or_target]
            return X, y
        else:
            return X_train_or_data, y_test_or_target
