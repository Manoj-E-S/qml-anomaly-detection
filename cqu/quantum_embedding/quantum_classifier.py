import time
from typing import List, overload

import numpy as np
import pandas as pd
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
        self, test_size: float, random_state: int, backend_provider: str
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

        self.feature_map = None
        self.var_form = None
        self.full_circuit = None
        self.opt_var = None

        self.test_size = test_size
        self.random_state = random_state

        self.__initialize_circuit()

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

    def __initialize_circuit(self) -> None:
        self.feature_map = ZZFeatureMap(feature_dimension=5, reps=2)
        self.feature_map.barrier()
        self.var_form = TwoLocal(5, ["ry", "rz"], "cz", reps=3)

        self.full_circuit = self.feature_map.compose(self.var_form)
        self.full_circuit.measure_all()

        self.optimizer = SPSA(maxiter=50)
