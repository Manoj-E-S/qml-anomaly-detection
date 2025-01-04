# from cqu.quantum_embedding import QuantumClassifier
from sklearn.model_selection import train_test_split

from cqu.preprocessing import Preprocessor

print("Using fraud dataset")
cqp = Preprocessor("./datasets/ccfraud/creditcard.csv")

selected_features = ["v17", "v12", "v14", "v16", "v10", "class"]
dataset = cqp.dataframe[selected_features]

# Print row count of each class
print("Class distribution in original dataset:")
print(dataset["class"].value_counts())

import pandas as pd

total_rows = 1000
fraud_rows = 100
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


from qiskit_ibm_runtime import QiskitRuntimeService

# QiskitRuntimeService.save_account(
#     channel="ibm_quantum",
#     token="4bccb8d54705ea83a2c6b462ec4cdbda2162bafbd46c2112975f2397fa24ba37eb56e08e1c99ba48b0b9d289c776414bb7a59bb435c0df65fadb21e46821bb17"
# )

service = QiskitRuntimeService()
backend = service.least_busy()
print("Using backend: ", backend)

# print("Testing QuantumClassifier")
# qc = QuantumClassifier()
# qc.train(dataset, 'class')

# import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.optimizers import SPSA

features = X.shape[1]
random_state = 42
shots = 1024

feature_map = ZZFeatureMap(feature_dimension=features, reps=2)
feature_map.barrier()
var_form = TwoLocal(features, ["ry", "rz"], "cz", reps=2)

full_circuit = QuantumCircuit(features)
full_circuit.append(feature_map, range(features))
full_circuit.append(var_form, range(features))

# from qiskit.quantum_info import PauliList
# import numpy as np

# # Create a cost Hamiltonian for classification (adjust for your dataset)
# cost_pauli_list = PauliList(["Z" * features])

# from qiskit import transpile

# i = 0

# # Define a function to evaluate the cost
# def evaluate_cost(params, circuit, labels, pauli_list):
#     global i, backend
#     print(f"Running evaluation {i}")
#     i += 1

#     param_dict = {param: val for param, val in zip(circuit.parameters, params)}

#     # Bind parameters to the circuit
#     bound_circuit = circuit.assign_parameters(param_dict)

#     # Transpile the circuit for the backend
#     transpiled_circuit = transpile(bound_circuit, backend)

#     # Execute the circuit
#     job = backend.run(transpiled_circuit, shots=1024)
#     counts = job.result().get_counts()

#     # Convert counts to probabilities
#     shots = sum(counts.values())
#     probabilities = {state: count / shots for state, count in counts.items()}

#     # Compute the expectation value for each Pauli in the list
#     expectation_value = 0
#     for pauli in pauli_list:
#         z_mask = [i for i, p in enumerate(pauli) if p == 'Z']

#         # Compute the expectation for this Pauli term
#         pauli_expectation = sum(
#             prob * (-1 if sum(int(state[q]) for q in z_mask) % 2 else 1)
#             for state, prob in probabilities.items()
#         )

#         expectation_value += pauli_expectation

#     # Simplified cost based on labels
#     # For binary classification, the cost could involve the overlap with labels
#     cost = np.mean([expectation_value * label for label in labels])
#     return cost


# print("Optimizing the circuit using SPSA")

# optimizer = SPSA(maxiter=100)
# initial_point = np.zeros((var_form.num_parameters))

# def objective_function(params):
#     return evaluate_cost(params, full_circuit, y_train, cost_pauli_list)

# result = optimizer.minimize(objective_function, initial_point)

# optimal_params = result.x
# print("Optimized parameters:", optimal_params)
# print("Final cost:", result.fun)

# def circuit_instance(data, variational):
#     """Assigns parameter values to `AD_HOC_CIRCUIT`.
#     Args:
#         data (list): Data values for the feature map
#         variational (list): Parameter values for `VAR_FORM`
#     Returns:
#         QuantumCircuit: `AD_HOC_CIRCUIT` with parameters assigned
#     """
#     parameters = {}
#     for i, p in enumerate(feature_map.ordered_parameters):
#         parameters[p] = data[i]
#     for i, p in enumerate(var_form.ordered_parameters):
#         parameters[p] = variational[i]
#     return ad_hoc_circuit.assign_parameters(parameters)

# def parity(bitstring):
#     """Returns 1 if parity of `bitstring` is even, otherwise 0."""
#     hamming_weight = sum(int(k) for k in list(bitstring))
#     return (hamming_weight+1) % 2

# def label_probability(results):
#     """Converts a dict of bitstrings and their counts,
#     to parities and their counts"""
#     shots = sum(results.values())
#     probabilities = {0: 0, 1: 0}
#     for bitstring, counts in results.items():
#         label = parity(bitstring)
#         probabilities[label] += counts / shots
#     return probabilities


# def classification_probability(data, variational):
#     """Classify data points using given parameters.
#     Args:
#         data (list): Set of data points to classify
#         variational (list): Parameters for `VAR_FORM`
#     Returns:
#         list[dict]: Probability of circuit classifying
#                     each data point as 0 or 1.
#     """
#     circuits = [circuit_instance(d, variational) for d in data]
#     sampler = StatevectorSampler()
#     results = sampler.run(circuits).result()
#     classification = [
#         label_probability(results[i].data.meas.get_counts()) for i, c in enumerate(circuits)]
#     return classification


# def cross_entropy_loss(classification, expected):
#     """Calculate accuracy of predictions using cross entropy loss.
#     Args:
#         classification (dict): Dict where keys are possible classes,
#                                and values are the probability our
#                                circuit chooses that class.
#         expected (int): Correct classification of the data point.

#     Returns:
#         float: Cross entropy loss
#     """
#     p = classification.get(expected)  # Prob. of correct classification
#     return -np.log(p + 1e-10)


# def cost_function(data, labels, variational):
#     """Evaluates performance of our circuit with `variational`
#     parameters on `data`.

#     Args:
#         data (list): List of data points to classify
#         labels (list): List of correct labels for each data point
#         variational (list): Parameters to use in circuit

#     Returns:
#         float: Cost (metric of performance)
#     """
#     classifications = classification_probability(data, variational)
#     cost = 0
#     for i, classification in enumerate(classifications):
#         cost += cross_entropy_loss(classification, labels[i])
#     cost /= len(data)
#     return cost

# def objective_function(variational):
#     """Cost function of circuit parameters on training data.
#     The optimizer will attempt to minimize this."""
#     return cost_function(X_train, y_train, variational)

# print("Optimizing the circuit running minimizer")
# result = optimizer.minimize(objective_function, initial_point)


# opt_var = result.x
# opt_value = result.fun

# print("Optimal parameters: ", opt_var)
# print("Optimal cost: ", opt_value)

# inputs = {
#     'ansatz': full_circuit,
#     'optimizer': {'name': 'SPSA', 'maxiter': 50},
#     'initial_point': initial_point,
#     'shots': 1024
# }

# options = {
#     'backend_name': 'ibm_brisbane',
# }

# job = service.run(program_id='vqe', options=options, inputs=inputs)

# result = job.result()
# print("Optimal parameters:", result['optimal_point'])
# print("Final cost:", result['optimal_value'])

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_ibm_runtime import Batch, SamplerV2
from qiskit_machine_learning.algorithms import VQC, NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import SamplerQNN

# features = X.shape[1]

# feature_map = ZZFeatureMap(feature_dimension=features, reps=2)
# feature_map.barrier()
# var_form = TwoLocal(features, ['ry', 'rz'], 'cz', reps=2)

# full_circuit = QuantumCircuit(features)
# full_circuit.append(feature_map, range(features))
# full_circuit.append(var_form, range(features))

batch = Batch(backend=backend)
sampler = SamplerV2(mode=batch)

# input_params = [Parameter(f'Î¸{i}') for i in range(5)]
# weight_params = [Parameter(f'w{i}') for i in range(3)]

# print("Creating Quantum Circuit")
# def create_quantum_circuit():
#     """Create a parameterized quantum circuit for 5 input parameters."""
#     qr = QuantumRegister(6, 'qr')
#     cr = ClassicalRegister(1, 'cr')
#     qc = QuantumCircuit(qr, cr)

#     # Encode input parameters
#     for i, theta in enumerate(input_params):
#         qc.ry(theta, qr[i])

#     # Create entanglement layers
#     # First entanglement layer
#     for i in range(4):
#         qc.cx(qr[i], qr[i+1])

#     # Trainable rotation layer
#     qc.ry(weight_params[0], qr[0])
#     qc.ry(weight_params[1], qr[2])
#     qc.ry(weight_params[2], qr[4])

#     # Second entanglement layer
#     for i in range(0, 4, 2):
#         qc.cx(qr[i], qr[i+1])

#     # Final measurement on the last qubit
#     qc.measure(qr[5], cr[0])

#     return qc

# # Create training data
# num_samples = 20

# print("Creating Quantum Neural Network")
# Create the quantum neural network
# circuit = create_quantum_circuit()


print("Transpile the circuit")
# Transpile the circuit
# transpiled_circuit = transpile(full_circuit, backend)
transpiled_circuit = transpile(full_circuit, backend)

vqc = VQC(ansatz=transpiled_circuit, optimizer=SPSA(maxiter=50), sampler=sampler)

# qnn = SamplerQNN(
#     circuit=transpiled_circuit,
#     input_params=input_params,
#     weight_params=weight_params,
#     output_shape=2,
#     sampler=sampler
# )

# print("Creating and training the classifier")
# # Create and train the classifier
# optimizer = COBYLA(maxiter=50)
# classifier = NeuralNetworkClassifier(
#     neural_network=qnn,
#     optimizer=optimizer
# )

print("Fitting the classifier")
# Train the model
# classifier.fit(X_train, y_train)
vqc.fit(X_train, y_train)

print("Evaluating the model")
# y_pred = classifier.predict(X_test)
y_pred = vqc.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_model(y_true, y_pred):
    """Calculate and print all evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


evaluate_model(y_test, y_pred)

batch.close()


def predict_fraud(transaction_data):
    """Predict if a transaction is fraudulent."""
    # prediction = classifier.predict(transaction_data)
    prediction = vqc.predict(transaction_data)
    return prediction
