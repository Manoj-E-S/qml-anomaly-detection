from numpy.typing import ArrayLike

from cqu.typing import Dataset
from cqu.utils import BaseClassifier, ClassifierMetrics, get_metrics


class QuantumSVM(BaseClassifier):
    def fit(self, X_train_or_data: Dataset, y_train_or_target: ArrayLike | str) -> None:
        pass

    def test(
        self, X_train_or_data: Dataset, y_test_or_target: ArrayLike | str
    ) -> ClassifierMetrics:
        pass

    def predict(self, X: Dataset) -> ArrayLike:
        pass
