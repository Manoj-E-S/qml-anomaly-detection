from time import time

import pandas as pd
from sklearn.model_selection import train_test_split

from cqu.preprocessing import Preprocessor
from cqu.quantum import QuantumSVM


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


random_state = 42

print("Loading dataset...")
cqp = Preprocessor("./datasets/ccfraud/creditcard.csv")

df = reduce_dataset(cqp.dataframe, 100, 10)
df = df[["v17", "v12", "v14", "v16", "v10", "class"]]

X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

print("Creating model...")

qsvm = QuantumSVM(
    num_features=X_train.shape[1],
    class_weight={0: 1, 1: 100},
    random_state=random_state,
)

print("Training model...")

start_time = time()
qsvm.fit(X_train, y_train)

print(f"Training took {time() - start_time:.2f} seconds")

print("Testing...")

start_time = time()
metrics = qsvm.test(X_test, y_test)

print(f"Testing took {time() - start_time:.2f} seconds")

print(metrics)
