import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from cqu.classical import (
    ClassicalModels,
    get_feature_importances,
    get_the_best_classical_model,
)
from cqu.preprocessing import Preprocessor
from cqu.utils import PLOT_FOLDER_NAME
from cqu.utils.metrics import ClassifierMetrics
from cqu.utils.plotting import plot_all_metrics


def get_dataset(path: str) -> pd.DataFrame:
    cqp = Preprocessor(path)

    selected_features = ["v17", "v12", "v14", "v16", "v10", "class"]
    dataset = cqp.dataframe[selected_features]

    # Print row count of each class
    print("Class distribution in original dataset:")
    print(dataset["class"].value_counts())

    return dataset


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


def log_model_metrics(metrics: ClassifierMetrics) -> None:
    path = os.path.join(
        PLOT_FOLDER_NAME, f"{metrics.model_type}_evaluation_results.txt"
    )
    with open(path, "w") as f:
        f.write(metrics.to_string())


should_plot = True


def log_and_plot(metrics: ClassifierMetrics) -> None:
    log_model_metrics(metrics)
    if should_plot:
        plot_all_metrics(metrics)


if __name__ == "__main__":
    print("Getting fraud dataset")
    df = get_dataset("./datasets/ccfraud/creditcard.csv")
    # df = reduce_dataset(df, 100, 10)
    target_column = "class"
    parallel = False

    print("Getting feature importances")

    top_n = 5
    feature_importances = get_feature_importances(
        {
            ClassicalModels.LOGISTIC_REGRESSION: top_n,
            ClassicalModels.RANDOM_FOREST: top_n,
            ClassicalModels.GRADIENT_BOOSTING: top_n,
            ClassicalModels.NEURAL_NETWORK: top_n,
            ClassicalModels.KNN: top_n,
            ClassicalModels.NAIVE_BAYES: top_n,
        },
        df,
        target_column,
    )

    for model, data in feature_importances.items():
        features = data["feature"]
        print(f"{model}\n = {features}")

    best_model, best_f1_score = get_the_best_classical_model(
        df, target_column, feature_importances, parallel=True
    )
    print(f"Best Model: {best_model} | F1-Score: {best_f1_score}")
