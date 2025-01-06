from typing import Dict

from numpy.typing import ArrayLike
from sklearn.svm import SVC

from cqu.typing import Dataset, QuantumModels
from cqu.utils.classifier import BaseClassifier, _optimize_threshold
from cqu.utils.metrics import ClassifierMetrics, get_metrics


class QuantumSVM(BaseClassifier):
    qsvm: SVC

    def __init__(
        self,
        class_weight: Dict[int | str, int] | str | None = None,
        random_state=42,
        test_size=0.2,
    ):
        qkernel = self.get_quantum_kernel()
        self.qsvm = SVC(
            kernel=qkernel,
            probability=True,
            class_weight=class_weight,
            random_state=random_state,
        )

        super().__init__(random_state, test_size)

    def fit(self, X_train_or_data: Dataset, y_train_or_target: ArrayLike | str) -> None:
        X_train, y_train = self.__handle_data_split(X_train_or_data, y_train_or_target)

        self.qsvm.fit(X_train, y_train)

    def test(
        self, X_test_or_data: Dataset, y_test_or_target: ArrayLike | str
    ) -> ClassifierMetrics:
        X_test, y_test = self.__handle_data_split(X_test_or_data, y_test_or_target)

        y_proba = self.qsvm.predict_proba(X_test)[:, 1]

        threshold = _optimize_threshold(y_proba=y_proba, y_test=y_test)
        y_pred = (y_proba >= threshold).astype(int)

        return get_metrics(QuantumModels.QSVM, y_test, y_pred)

    def predict(self, X: Dataset) -> ArrayLike:
        pass

    def get_quantum_kernel(self):
        pass
