import numpy as np
import pandas as pd

from cqu.preprocessing import Preprocessor
from cqu.quantum import QuantumClassifier

# Preprocessing

print("Loading fraud dataset")
cqp = Preprocessor("./datasets/ccfraud/creditcard.csv")

selected_features = ["v17", "v12", "v14", "v16", "v10", "class"]
dataset = cqp.dataframe[selected_features]

# Print row count of each class
print("Class distribution in original dataset:")
print(dataset["class"].value_counts())

total_rows = 500
fraud_rows = 50
non_fraud_rows = total_rows - fraud_rows

fraud_data = dataset[dataset["class"] == 1].sample(n=fraud_rows, random_state=42)
non_fraud_data = dataset[dataset["class"] == 0].sample(
    n=non_fraud_rows, random_state=42
)
dataset = pd.concat([fraud_data, non_fraud_data])

from sklearn.model_selection import train_test_split

y = dataset["class"]
X = dataset.drop("class", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Class distribution in reduced dataset:")
print(dataset["class"].value_counts())

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel

print("Creating feature map and quantum kernel")
feature_map = ZZFeatureMap(feature_dimension=5, reps=2)

quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

import time

start_time = time.time()
print("Training quantum SVC")
qsvc = QSVC(quantum_kernel=quantum_kernel)
qsvc.fit(X_train, y_train)
print(
    "Training time: ",
    time.time() - start_time,
    "Minutes: ",
    (time.time() - start_time) / 60,
)


print("Predicting")
y_pred = qsvc.predict(X_test)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

print("Metrics")
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("Confusion matrix: \n", confusion_matrix(y_test, y_pred))
