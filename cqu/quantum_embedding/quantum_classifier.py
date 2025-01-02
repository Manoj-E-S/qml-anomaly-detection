import time
from typing import List, overload

import numpy as np
import pandas as pd
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import SPSA
from qiskit_ibm_runtime import Batch, SamplerV2
from qiskit_ibm_runtime.qiskit_runtime_service import Backend, QiskitRuntimeService
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


class QuantumClassifier:
    use_backend: bool
    service: QiskitRuntimeService | None
    backend: Backend | None
    batch: Batch | None

    num_features: int
    feature_map: ZZFeatureMap | None
    var_form: TwoLocal | None
    full_circuit: QuantumCircuit | None
    opt_var: List[float] | None

    optimizer: SPSA | None

    test_size: float
    random_state: int

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, test_size: float, random_state: int) -> None: ...

    @overload
    def __init__(
        self, test_size: float, random_state: int, backend_token: str
    ) -> None: ...

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        backend_token: str | None = None,
    ) -> None:
        self.use_backend = backend_token is not None
        self.service = (
            QiskitRuntimeService(channel="ibm_quantum", token=backend_token)
            if self.use_backend
            else None
        )
        self.backend = self.service.least_busy() if self.use_backend else AerSimulator()
        self.batch = Batch(backend=self.backend)

        self.num_features = 0
        self.feature_map = None
        self.var_form = None
        self.full_circuit = None
        self.opt_var = None

        self.test_size = test_size
        self.random_state = random_state

    @overload
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> None: ...

    @overload
    def train(self, dataset: pd.DataFrame, class_column_name: str) -> None: ...

    def train(
        self, dataset_or_x: pd.DataFrame, y_or_class_column_name: pd.DataFrame | str
    ) -> None:
        X = None
        y = None

        if isinstance(y_or_class_column_name, str):
            y = dataset_or_x[y_or_class_column_name].values
            X = dataset_or_x.drop(columns=[y_or_class_column_name]).values
        else:
            y = y_or_class_column_name
            X = dataset_or_x

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        self.num_features = X_train.shape[1]
        self.__initialize_circuit()

        def objective_function(variational):
            nonlocal X_train, y_train
            return self.__cost_function(X_train, y_train, variational)

        initial_point = np.zeros((self.var_form.num_parameters))

        # Time the minimization
        start_time = time.time()

        result = self.optimizer.minimize(objective_function, initial_point)
        self.opt_var = result.x

        training_time = time.time() - start_time

        start_time = time.time()

        probability = self.__classification_probability(X_test, self.opt_var)
        predictions = [0 if p[0] >= p[1] else 1 for p in probability]

        testing_time = time.time() - start_time

        print("###################################")
        print("Training Results: ")
        print(
            f"Training Time: Seconds: {training_time}, Minutes: {training_time / 60.0}"
        )
        print(f"Testing Time: Seconds: {testing_time}, Minutes: {testing_time / 60.0}")
        print("Accuracy: ", accuracy_score(y_test, predictions))
        print("Precision: ", precision_score(y_test, predictions))
        print("Recall: ", recall_score(y_test, predictions))
        print("F1 Score: ", f1_score(y_test, predictions))
        print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
        print("###################################")

    def predict(self, data: pd.DataFrame) -> List[int]:
        return [
            0 if p[0] >= p[1] else 1
            for p in self.__classification_probability(data.values, self.opt_var)
        ]

    def __circuit_instance(
        self, data: np.ndarray, variational: np.ndarray
    ) -> QuantumCircuit:
        parameters = {}
        for i, p in enumerate(self.feature_map.ordered_parameters):
            parameters[p] = data[i]
        for i, p in enumerate(self.var_form.ordered_parameters):
            parameters[p] = variational[i]
        return self.full_circuit.assign_parameters(parameters)

    def __parity(self, bitstring: str) -> int:
        hamming_weight = sum(int(k) for k in list(bitstring))
        return (hamming_weight + 1) % 2

    def __label_probability(self, results: dict) -> dict:
        shots = sum(results.values())
        probabilities = {0: 0, 1: 0}
        for bitstring, counts in results.items():
            label = self.__parity(bitstring)
            probabilities[label] += counts / shots
        return probabilities

    def __classification_probability(
        self, data: np.ndarray, variational: np.ndarray
    ) -> List[dict]:
        circuits = [
            transpile(self.__circuit_instance(d, variational), backend=self.backend)
            for d in data
        ]
        sampler = SamplerV2(mode=self.batch)
        results = sampler.run(circuits).result()
        classification = [
            self.__label_probability(results[i].data.meas.get_counts())
            for i, c in enumerate(circuits)
        ]
        return classification

    def __cross_entropy_loss(self, classification: dict, expected: int) -> float:
        p = classification.get(expected)
        return -np.log(p + 1e-10)

    def __cost_function(
        self, X: np.ndarray, y: np.ndarray, variational: np.ndarray
    ) -> float:
        classifications = self.__classification_probability(X, variational)
        cost = 0
        for i, classification in enumerate(classifications):
            cost += self.__cross_entropy_loss(classification, y[i])
        cost /= len(y)
        return cost

    def __initialize_circuit(self) -> None:
        self.feature_map = ZZFeatureMap(feature_dimension=self.num_features, reps=2)
        self.feature_map.barrier()
        self.var_form = TwoLocal(self.num_features, ["ry", "rz"], "cz", reps=2)

        self.full_circuit = self.feature_map.compose(self.var_form)
        self.full_circuit.measure_all()

        self.optimizer = SPSA(maxiter=50)
