from typing import List, overload

import pandas as pd
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import SPSA
from qiskit_ibm_runtime import Batch, SamplerV2
from qiskit_ibm_runtime.qiskit_runtime_service import Backend, QiskitRuntimeService
from sklearn.model_selection import train_test_split


class QuantumClassifier:
    use_backend: bool
    service: QiskitRuntimeService | None
    backend: Backend | None
    batch: Batch | None
    feature_map: ZZFeatureMap | None
    variational: TwoLocal | None
    full_circuit: QuantumCircuit | None
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
        self.variational = None
        self.full_circuit = None

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

    def __initialize_circuit(self) -> None:
        self.feature_map = ZZFeatureMap(feature_dimension=5, reps=2)
        self.feature_map.barrier()
        self.variational = TwoLocal(5, ["ry", "rz"], "cz", reps=3)
