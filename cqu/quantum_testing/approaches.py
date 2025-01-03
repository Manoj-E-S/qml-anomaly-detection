from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import SPSA
from qiskit.circuit import Parameter

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import SPSA

def create_hardware_compatible_feature_map(num_features):
    """
    Creates a hardware-compatible feature mapping circuit that matches
    the input data dimensions exactly.
    
    Args:
        num_features: Number of features in the input data
    """
    # Create circuit with same number of qubits as features
    # We don't need the +1 here since VQC will handle the classification qubit
    circuit = QuantumCircuit(num_features)
    
    # Encode features into quantum states
    for i in range(num_features):
        # Use RY and RZ gates for amplitude and phase encoding
        circuit.ry(Parameter(f'x_{i}'), i)
        circuit.rz(Parameter(f'x_{i}_phase'), i)
    
    # Create entanglement between adjacent qubits
    for i in range(num_features - 1):
        circuit.cx(i, i + 1)
    
    # Add second rotation layer for more expressiveness
    for i in range(num_features):
        circuit.ry(Parameter(f'x_{i}_second'), i)
    
    return circuit


def create_variational_circuit(num_features):
    """
    Creates a variational circuit for learning the classification boundary.
    This circuit matches the feature map dimensions.
    
    Args:
        num_features: Number of features in the input data
    """
    circuit = QuantumCircuit(num_features)
    
    # First variational layer
    for i in range(num_features):
        circuit.ry(Parameter(f'θ_{i}'), i)
        circuit.rz(Parameter(f'φ_{i}'), i)
    
    # Entangling layer using CNOT gates
    for i in range(num_features - 1):
        circuit.cx(i, i + 1)
    
    # Second variational layer
    for i in range(num_features):
        circuit.ry(Parameter(f'θ_{i}_2'), i)
        circuit.rz(Parameter(f'φ_{i}_2'), i)
    
    return circuit


def create_fraud_detection_circuit(num_features):
    """
    Combines feature mapping and Born machine circuits into a complete
    fraud detection quantum circuit. Both subcircuits now have the same
    number of qubits (num_features + 1).
    
    Args:
        num_features: Number of features in the input data
        
    Returns:
        QuantumCircuit: Complete fraud detection circuit
    """
    # Create the complete circuit with an extra qubit for classification
    circuit = QuantumCircuit(num_features + 1)
    
    # Add feature mapping circuit
    feature_map = create_hardware_compatible_feature_map(num_features)
    circuit.compose(feature_map, inplace=True)
    
    # Add barrier for clarity in visualization
    circuit.barrier()
    
    # Add Born machine circuit
    var_circuit = create_variational_circuit(num_features)
    circuit.compose(var_circuit, inplace=True)
    
    # Add barrier before measurement
    circuit.barrier()
    
    # Add measurement only to the last qubit (classification qubit)
    circuit.measure_all()
    
    return circuit


# def create_circuit_with_unique_params(num_features):
#     """
#     Creates a quantum circuit with uniquely named parameters to avoid conflicts.
#     Uses a feature map for data encoding and variational form for learning.
#     """
#     # Create feature map with explicit parameter naming
#     feature_map = ZZFeatureMap(
#         feature_dimension=num_features,
#         reps=2,
#         parameter_prefix='fm'  # This ensures feature map parameters start with 'fm'
#     )
    
#     # Create variational form with different parameter prefix
#     var_form = TwoLocal(
#         num_features,
#         ['ry', 'rz'],
#         'cz',
#         reps=2,
#         parameter_prefix='vf'  # This ensures variational form parameters start with 'vf'
#     )
    
#     # Combine circuits
#     circuit = QuantumCircuit(num_features)
#     circuit.compose(feature_map, inplace=True)
#     circuit.barrier()  # Add barrier for clarity
#     circuit.compose(var_form, inplace=True)
    
#     return circuit

# Alternative 1: Quantum Kernel Approach
# from qiskit_machine_learning.kernels import QuantumKernel
# from qiskit_machine_learning.algorithms import QSVC

# def quantum_kernel_classifier(X_train, y_train, X_test, sampler):
#     """
#     Implements quantum kernel-based classification using QSVC.
#     This approach maps data to quantum feature space using kernel methods.
#     """
#     # Create quantum kernel
#     quantum_kernel = QuantumKernel(
#         feature_dimension=X_train.shape[1],
#         sampler=sampler
#     )
    
#     # Create and train quantum SVM
#     qsvc = QSVC(quantum_kernel=quantum_kernel)
#     qsvc.fit(X_train, y_train)
    
#     return qsvc

# Alternative 2: Quantum Circuit Born Machine
def create_born_machine_circuit(num_features):
    """
    Creates a Born Machine circuit for generative modeling of fraud patterns.
    This approach learns the probability distribution of legitimate vs fraudulent transactions.
    """
    qc = QuantumCircuit(num_features + 1)  # Extra qubit for classification
    
    # Initial superposition
    qc.h(range(num_features + 1))
    
    # Feature encoding
    for i in range(num_features):
        qc.ry(Parameter(f'data_{i}'), i)
    
    # Entangling layers
    for i in range(num_features):
        qc.cx(i, (i + 1) % (num_features + 1))
    
    # Measurement basis rotation
    qc.ry(Parameter('theta'), num_features)
    
    return qc

# Alternative 3: Quantum Distance-Based Classifier
def create_distance_based_circuit(num_features):
    """
    Creates a circuit that measures quantum distance between input and reference states.
    Useful for detecting anomalous transactions that are "far" from legitimate patterns.
    """
    qc = QuantumCircuit(2 * num_features)  # Double qubits for state comparison
    
    # Encode input state in first register
    for i in range(num_features):
        qc.ry(Parameter(f'input_{i}'), i)
    
    # Encode reference state in second register
    for i in range(num_features):
        qc.ry(Parameter(f'ref_{i}'), i + num_features)
    
    # Compute quantum distance through SWAP test
    for i in range(num_features):
        qc.cswap(0, i, i + num_features)
    
    return qc


def get_dataset():
    from cqu.preprocessing import Preprocessor

    print("Using fraud dataset")
    cqp = Preprocessor("./datasets/ccfraud/creditcard.csv")

    selected_features = ['v17', 'v12', 'v14', 'v16', 'v10', 'class']
    dataset = cqp.dataframe[selected_features]

    # Print row count of each class
    print("Class distribution in original dataset:")
    print(dataset['class'].value_counts())

    return dataset


def reduce_dataset(dataset):
    import pandas as pd

    total_rows = 1000
    fraud_rows = 100
    non_fraud_rows = total_rows - fraud_rows

    fraud_data = dataset[dataset['class'] == 1].sample(n=fraud_rows, random_state=42)
    non_fraud_data = dataset[dataset['class'] == 0].sample(n=non_fraud_rows, random_state=42)
    dataset = pd.concat([fraud_data, non_fraud_data])

    print("Class distribution in reduced dataset:")
    print(dataset['class'].value_counts())

    return dataset


def setup_qiskit_runtime():
    from qiskit_ibm_runtime import QiskitRuntimeService

    # QiskitRuntimeService.save_account(
    #     channel="ibm_quantum", 
    #     token="4bccb8d54705ea83a2c6b462ec4cdbda2162bafbd46c2112975f2397fa24ba37eb56e08e1c99ba48b0b9d289c776414bb7a59bb435c0df65fadb21e46821bb17"
    # )

    service = QiskitRuntimeService()
    backend = service.least_busy()
    print("Using backend: ", backend)

    return backend


def quantum_fraud_detector(X_train, y_train, sampler):
    """
    Creates and trains a quantum fraud detection model using
    hardware-compatible circuits.
    
    Args:
        X_train: Training features
        y_train: Training labels
        sampler: Qiskit sampler instance
    
    Returns:
        Trained VQC model
    """
    num_features = X_train.shape[1]
    
    # Create the hardware-compatible circuit
    circuit = create_fraud_detection_circuit(num_features)
    
    # Create the VQC with appropriate optimizer
    vqc = VQC(
        ansatz=circuit,
        optimizer=SPSA(maxiter=50),
        sampler=sampler
    )
    
    # Train the model
    vqc.fit(X_train, y_train)
    
    return vqc

# Example usage
def run_fraud_detection(X_train, y_train, X_test, sampler):
    """
    Complete pipeline for quantum fraud detection.
    """
    # Train the model
    model = quantum_fraud_detector(X_train, y_train, sampler)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    return predictions, model


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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


if __name__ == "__main__":
    ds = get_dataset()
    ds = reduce_dataset(ds)

    from sklearn.model_selection import train_test_split
    y = ds['class'].values
    X = ds.drop(columns=['class']).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    num_features = X_train.shape[1]

    from qiskit_ibm_runtime import SamplerV2, Batch
    backend = setup_qiskit_runtime()
    batch = Batch(backend=backend)
    sampler = SamplerV2(mode=batch)

    # Create and train the model
    y_pred, trained_model = run_fraud_detection(X_train, y_train, X_test, sampler)

    # Evaluate the results
    evaluate_model(y_test, y_pred)

    batch.close()


