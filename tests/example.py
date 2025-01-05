import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from cqu.preprocessing import Preprocessor
from cqu.quantum import QuantumClassifier

# Preprocessing

print("Loading fraud dataset")
cqp = Preprocessor("./datasets/ccfraud/creditcard.csv")

dataset = cqp.dataframe[["v10", "v12", "v14", "v16", "v17", "class"]]

print("Class distribution in original dataset:")
print(dataset["class"].value_counts())

total_rows = 100
fraud_rows = 40
non_fraud_rows = total_rows - fraud_rows

fraud_data = dataset[dataset["class"] == 1].sample(n=fraud_rows, random_state=42)
non_fraud_data = dataset[dataset["class"] == 0].sample(
    n=non_fraud_rows, random_state=42
)
dataset = pd.concat([fraud_data, non_fraud_data])

print("Row Count: ", dataset.shape[0])

print("Class distribution in reduced dataset:")
print(dataset["class"].value_counts())

y = dataset["class"]
X = dataset.drop("class", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import StandardScaler

# Scale

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Quantum Feature Importance

from qiskit.circuit.library import PauliFeatureMap, ZZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute

print("Shape: ", X.shape[1])

backend = AerSimulator()
sampler = SamplerV2(mode=backend)

# feature_map = PauliFeatureMap(feature_dimension=X.shape[1], reps=4, paulis=['Z', 'ZZ'])
feature_map = ZZFeatureMap(
    feature_dimension=X.shape[1], reps=4, entanglement="linear", insert_barriers=True
)
fidelity = ComputeUncompute(sampler=sampler)
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

# Try first with classical

from sklearn.svm import SVC

print("Training classical SVC")

kernel_matrix_train = quantum_kernel.evaluate(X_train)
kernel_matrix_test = quantum_kernel.evaluate(X_test, X_train)

svc = SVC(kernel="precomputed")
svc.fit(kernel_matrix_train, y_train)
y_pred_svc = svc.predict(kernel_matrix_test)

print("Classical SVM Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_svc))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svc))

import time

print("Training quantum SVC")
start_time = time.time()
qsvc = QSVC(quantum_kernel=quantum_kernel)
qsvc.fit(X_train, y_train)
print(
    "Training time: ",
    time.time() - start_time,
    "Minutes: ",
    (time.time() - start_time) / 60,
)

print("Predicting")
start_time = time.time()
y_pred = qsvc.predict(X_test)
print(
    "Predicting time: ",
    time.time() - start_time,
    "Minutes: ",
    (time.time() - start_time) / 60,
)

print("Metrics")
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("Confusion matrix: \n", confusion_matrix(y_test, y_pred))

# # Training

# print("Training quantum classifier")

# qc = QuantumClassifier()
# qc.train(dataset, "class")

# # Prediction

# test_df = pd.DataFrame(np.random.rand(5))
# prediction = qc.predict(test_df)

# if prediction[0] == 1:
#     print("Prediction: Fraud")
# else:
#     print("Prediction: Non-fraud")
