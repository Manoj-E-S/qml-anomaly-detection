from typing import Callable, Dict

import numpy as np
import pennylane as pln
from numpy.typing import ArrayLike
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC

from cqu.typing import Dataset, QuantumModels
from cqu.utils.classifier import BaseClassifier, _optimize_threshold
from cqu.utils.metrics import ClassifierMetrics, get_metrics


class QuantumSVM(BaseClassifier):
    num_features: int
    threshold: float | None
    scaler: MaxAbsScaler
    qsvm: SVC

    def __init__(
        self,
        num_features: int,
        class_weight: Dict[int | str, int] | str | None = None,
        random_state: int = 42,
        test_size: float = 0.2,
        custom_kernal: Callable[[ArrayLike, ArrayLike], ArrayLike] | None = None,
    ):
        self.num_features = num_features
        self.threshold = None
        self.scaler = MaxAbsScaler()
        qkernel = (
            custom_kernal
            if custom_kernal is not None
            else self.__get_quantum_kernel(num_qubits=self.num_features)
        )
        self.qsvm = SVC(
            kernel=qkernel,
            probability=True,
            class_weight=class_weight,
            random_state=random_state,
        )

        super().__init__(random_state, test_size)

    def fit(self, X_train_or_data: Dataset, y_train_or_target: ArrayLike | str) -> None:
        X_train, y_train = self._handle_data_split(X_train_or_data, y_train_or_target)
        self.__check_invalid_feature_count(X_train)
        X_train = self.scaler.fit_transform(X_train)

        self.qsvm.fit(X_train, y_train)

    def test(
        self, X_test_or_data: Dataset, y_test_or_target: ArrayLike | str
    ) -> ClassifierMetrics:
        X_test, y_test = self._handle_data_split(X_test_or_data, y_test_or_target)
        self.__check_invalid_feature_count(X_test)
        X_test = self.scaler.transform(X_test)

        y_proba = self.qsvm.predict_proba(X_test)[:, 1]
        self.threshold = _optimize_threshold(y_proba=y_proba, y_test=y_test)
        y_pred = (y_proba >= self.threshold).astype(int)

        return get_metrics(QuantumModels.QSVM, y_test, y_pred)

    def predict(self, X: Dataset) -> ArrayLike:
        if self.threshold is None:
            raise ValueError(
                "Threshold is not set. Please call fit() before calling predict()"
            )

        self.__check_invalid_feature_count(X)
        X = self.scaler.transform(X)

        y_proba = self.qsvm.predict_proba(X)[:, 1]
        y_pred = (y_proba >= self.threshold).astype(int)

        return y_pred

    def reinitialize_kernel(self, num_features: int):
        self.num_features = num_features
        self.qsvm.kernel = self.__get_quantum_kernel(num_qubits=self.num_features)

    def __get_quantum_kernel(self, num_qubits: int):
        device = pln.device("default.qubit", wires=num_qubits)

        @pln.qnode(device)
        def quantum_kernel_circuit(a, b):
            # Simple feature encoding
            for i in range(num_qubits):
                pln.RY(np.pi * a[i], wires=i)

            # Single entanglement layer
            for i in range(0, num_qubits - 1, 2):
                pln.CNOT(wires=[i, i + 1])

            # Inverse encoding of second feature vector
            for i in range(num_qubits):
                pln.RY(-np.pi * b[i], wires=i)

            return pln.probs(wires=[0])  # Measure only first qubit

        def kernel(A, B):
            A = np.asarray(A)
            B = np.asarray(B)

            kernel_matrix = np.zeros((len(A), len(B)))

            for i, a in enumerate(A):
                for j, b in enumerate(B):
                    kernel_matrix[i, j] = quantum_kernel_circuit(a, b)[0]

            return kernel_matrix

        return kernel

    def __check_invalid_feature_count(self, X: Dataset):
        if X.shape[1] != self.num_features:
            raise ValueError(
                f"Number of features in data ({X.shape[1]}) "
                f"does not match the number of features the model was initialzed with({self.num_features})"
                f"Make a call to reinitialize_kernel() to update the number of features or provide data with {self.num_features} features"
            )
