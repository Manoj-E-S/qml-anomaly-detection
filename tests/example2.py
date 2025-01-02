# Helper timer class
# ------------------

import time


class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            raise ValueError("Timer has not been started")

        duration = time.time() - self.start_time
        self.start_time = None

        return duration

    def time_function(self, name, callback, *args, **kwargs):
        self.start()
        callback(args, kwargs)
        duration = self.stop()
        print(f"{name} time: Seconds: {duration}, Minutes: {duration / 60.0}")


timer = Timer()

from sklearn.model_selection import train_test_split

# Data Preprocessing
# ------------------
from cqu.preprocessing import Preprocessor

print("Loading fraud dataset")
cqp = Preprocessor("./datasets/ccfraud/creditcard.csv")

selected_features = ["v17", "v12", "v14", "v16", "v10", "class"]
dataset = cqp.dataframe[selected_features]

# Print row count of each class
print("Class distribution in original dataset:")
print(dataset["class"].value_counts())

import pandas as pd

total_rows = 100
fraud_rows = 10
non_fraud_rows = total_rows - fraud_rows

fraud_data = dataset[dataset["class"] == 1].sample(n=fraud_rows, random_state=42)
non_fraud_data = dataset[dataset["class"] == 0].sample(
    n=non_fraud_rows, random_state=42
)
dataset = pd.concat([fraud_data, non_fraud_data])

print("Class distribution in reduced dataset:")
print(dataset["class"].value_counts())

y = dataset["class"].values
X = dataset.drop(columns=["class"]).values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Quantum Embedding
# -----------------

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import SPSA
from qiskit_ibm_runtime import Batch, SamplerV2
from qiskit_ibm_runtime.qiskit_runtime_service import QiskitRuntimeService

print("Getting AerSimulator backend")
# backend = AerSimulator()

service = QiskitRuntimeService(
    channel="ibm_quantum",
    token="a624b35041630eb3fc1f05eaea6c0929006b8a046c5c73f1f0d9100239033375082f5acb7730de3b3a4659b772a75b5e46d38f02ea53e398f260808de6045b5f",
)
backend = service.least_busy()

batch = Batch(backend=backend)

# Dataset metadata
feature_dim = 5
random_seed = 10598
shots = 1024

feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2)
feature_map.barrier()
var_form = TwoLocal(2, ["ry", "rz"], "cz", reps=2)

ad_hoc_circuit = feature_map.compose(var_form)
ad_hoc_circuit.measure_all()

# Some Stuff IDK


def circuit_instance(data, variational):
    """Assigns parameter values to `AD_HOC_CIRCUIT`.
    Args:
        data (list): Data values for the feature map
        variational (list): Parameter values for `VAR_FORM`
    Returns:
        QuantumCircuit: `AD_HOC_CIRCUIT` with parameters assigned
    """
    parameters = {}
    for i, p in enumerate(feature_map.ordered_parameters):
        parameters[p] = data[i]
    for i, p in enumerate(var_form.ordered_parameters):
        parameters[p] = variational[i]
    return ad_hoc_circuit.assign_parameters(parameters)


def parity(bitstring):
    """Returns 1 if parity of `bitstring` is even, otherwise 0."""
    hamming_weight = sum(int(k) for k in list(bitstring))
    return (hamming_weight + 1) % 2


def label_probability(results):
    """Converts a dict of bitstrings and their counts,
    to parities and their counts"""
    shots = sum(results.values())
    probabilities = {0: 0, 1: 0}
    for bitstring, counts in results.items():
        label = parity(bitstring)
        probabilities[label] += counts / shots
    return probabilities


def classification_probability(data, variational):
    """Classify data points using given parameters.
    Args:
        data (list): Set of data points to classify
        variational (list): Parameters for `VAR_FORM`
    Returns:
        list[dict]: Probability of circuit classifying
                    each data point as 0 or 1.
    """
    circuits = [
        transpile(circuit_instance(d, variational), backend=backend) for d in data
    ]
    sampler = SamplerV2(mode=batch)
    # sampler = StatevectorSampler()
    results = sampler.run(circuits).result()
    classification = [
        label_probability(results[i].data.meas.get_counts())
        for i, c in enumerate(circuits)
    ]
    return classification


def cross_entropy_loss(classification, expected):
    """Calculate accuracy of predictions using cross entropy loss.
    Args:
        classification (dict): Dict where keys are possible classes,
                               and values are the probability our
                               circuit chooses that class.
        expected (int): Correct classification of the data point.

    Returns:
        float: Cross entropy loss
    """
    p = classification.get(expected)  # Prob. of correct classification
    return -np.log(p + 1e-10)


def cost_function(data, labels, variational):
    """Evaluates performance of our circuit with `variational`
    parameters on `data`.

    Args:
        data (list): List of data points to classify
        labels (list): List of correct labels for each data point
        variational (list): Parameters to use in circuit

    Returns:
        float: Cost (metric of performance)
    """
    classifications = classification_probability(data, variational)
    cost = 0
    for i, classification in enumerate(classifications):
        cost += cross_entropy_loss(classification, labels[i])
    cost /= len(data)
    return cost


def objective_function(variational):
    """Cost function of circuit parameters on training data.
    The optimizer will attempt to minimize this."""
    return cost_function(X_train, y_train, variational)


print("Running optimizer")

timer.start()

optimizer = SPSA(maxiter=50)
initial_point = np.zeros((var_form.num_parameters))

result = optimizer.minimize(objective_function, initial_point)

opt_var = result.x
opt_value = result.fun

duration = timer.stop()
print(f"Training time: Seconds: {duration}, Minutes: {duration / 60.0}")

# Test classifier


def test_classifier(data, labels, variational):
    """Gets classifier's most likely predictions and accuracy of those
    predictions.

    Args:
        data (list): List of data points to classify
        labels (list): List of correct labels for each data point
        variational (list): List of parameter values for classifier

    Returns:
        float: Average accuracy of classifier over `data`
        list: Classifier's label predictions for each data point
    """
    probability = classification_probability(data, variational)
    predictions = [0 if p[0] >= p[1] else 1 for p in probability]
    accuracy = 0

    for i, prediction in enumerate(predictions):
        if prediction == labels[i]:
            accuracy += 1
    accuracy /= len(labels)
    return accuracy, predictions


print("Testing classifier")

timer.start()

accuracy, predictions = test_classifier(X_test, y_test, opt_var)

duration = timer.stop()
print(f"Evaluating time: Seconds: {duration}, Minutes: {duration / 60.0}")

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

print("Accuray by the classifier: ", accuracy)

print("Accuracy: ", accuracy_score(y_test, predictions))
print("Precision: ", precision_score(y_test, predictions))
print("Recall: ", recall_score(y_test, predictions))
print("F1 Score: ", f1_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

batch.close()
