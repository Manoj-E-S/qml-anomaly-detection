import numpy as np
import pandas as pd
import pennylane as pln
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC

from cqu.preprocessing import Preprocessor
from cqu.utils.metrics import get_metrics


def reduce_dataset(dataset: pd.DataFrame, total_rows, class_1_rows) -> pd.DataFrame:
    class_0_rows = total_rows - class_1_rows

    class_1_data = dataset[dataset["class"] == 1].sample(
        n=class_1_rows, random_state=42
    )
    class_0_data = dataset[dataset["class"] == 0].sample(
        n=class_0_rows, random_state=42
    )
    dataset = pd.concat([class_1_data, class_0_data])

    print("Class distribution in reduced dataset:")
    print(dataset["class"].value_counts())

    return dataset


print("Loading dataset...")
cqp = Preprocessor("./datasets/ccfraud/creditcard.csv")

df = reduce_dataset(cqp.dataframe, 1800, 200)
df = df[["v17", "v12", "v14", "v16", "v10", "class"]]

X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Scaling data...")
scaler = MaxAbsScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

num_features = X_train.shape[1]
num_qubits = num_features
import numpy as np
import pennylane as pln


def create_fast_kernel(num_qubits):
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
        # Convert to arrays if not already
        A = np.asarray(A)
        B = np.asarray(B)

        kernel_matrix = np.zeros((len(A), len(B)))

        # Compute kernel values with minimal transformations
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                kernel_matrix[i, j] = quantum_kernel_circuit(a, b)[0]

        return kernel_matrix

    return kernel


from time import time

print("Training model...")

start_time = time()
qkernel = create_fast_kernel(num_features)
qsvm = SVC(
    kernel=qkernel, probability=True, class_weight={0: 1, 1: 120}, random_state=42
)
qsvm.fit(X_train, y_train)
print(f"Training took {time() - start_time:.2f} seconds")

print("Tiling data for prediction...")
X_test = np.tile(X_test, (X_train.shape[0] // X_test.shape[0], 1))

print("Predicting...")
start_time = time()
y_proba = qsvm.predict_proba(X_test)[:, 1]
print(f"Prediction took {time() - start_time:.2f} seconds")

y_proba = y_proba[: len(y_test)]

from cqu.classical.models import _optimize_threshold

print("Optimizing threshold...")
threshold = _optimize_threshold(y_proba=y_proba, y_test=y_test)
y_pred = (y_proba >= threshold).astype(int)

print("Calculating metrics...")
print(get_metrics("QSVM", y_test, y_pred))
