from cqu.preprocessing import Preprocessor
import pandas as pd

def get_dataset(path:str) -> pd.DataFrame:
    cqp = Preprocessor(path)

    selected_features = ['v17', 'v12', 'v14', 'v16', 'v10', 'class']
    dataset = cqp.dataframe[selected_features]

    # Print row count of each class
    print("Class distribution in original dataset:")
    print(dataset['class'].value_counts())

    return dataset


def reduce_dataset(dataset:pd.DataFrame, total_rows, class_1_rows) -> pd.DataFrame:
    class_0_rows = total_rows - class_1_rows

    class_1_data = dataset[dataset['class'] == 1].sample(n=class_1_rows, random_state=42)
    class_0_data = dataset[dataset['class'] == 0].sample(n=class_0_rows, random_state=42)
    dataset = pd.concat([class_1_data, class_0_data])

    print("Class distribution in reduced dataset:")
    print(dataset['class'].value_counts())

    return dataset