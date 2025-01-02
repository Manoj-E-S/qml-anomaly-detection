import numpy as np
import pandas as pd

from cqu.preprocessing import Preprocessor
from cqu.quantum_embedding import QuantumClassifier

# Preprocessing

print("Loading fraud dataset")
cqp = Preprocessor("./datasets/ccfraud/creditcard.csv")

selected_features = ["v17", "v12", "v14", "v16", "v10", "class"]
dataset = cqp.dataframe[selected_features]

# Print row count of each class
print("Class distribution in original dataset:")
print(dataset["class"].value_counts())

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

qc = QuantumClassifier()
qc.train(dataset, "class")
prediction = qc.predict(np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]))

if prediction == 1:
    print("Prediction: Fraud")
else:
    print("Prediction: Non-fraud")
