import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from cqu.classical import ClassicalModels
from cqu.classical.important_features import get_feature_importances
from cqu.classical.models import (
    gradient_boosting_with_analysis,
    knn_model_with_analysis,
    logistic_regression_with_analysis,
    naive_bayes_model_with_analysis,
    neural_network_with_analysis,
    random_forest_with_analysis,
)
from cqu.preprocessing import Preprocessor
from cqu.utils import PLOT_FOLDER_NAME
from cqu.utils.metrics import ClassifierMetrics


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


def log_model_metrics(model_name, result: ClassifierMetrics) -> None:
    path = os.path.join(PLOT_FOLDER_NAME, f"{model_name}_evaluation_results.txt")
    with open(path, "w") as f:
        f.write(result.to_string())


if __name__ == "__main__":
    print("Getting fraud dataset")
    df = get_dataset("./datasets/ccfraud/creditcard.csv")
    df = reduce_dataset(df, 100, 10)
    target_column = "class"
    parallel = False

    if parallel == False:
        print("Executing Sequentially, and plotting")

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

        metrics = logistic_regression_with_analysis(
            df, target_column, feature_importances[ClassicalModels.LOGISTIC_REGRESSION]
        )
        log_model_metrics("Logistic Regression", metrics)

        # result = random_forest_with_analysis(
        #     df, target_column, feature_importances, shouldPlot=plotter.shouldPlot
        # )
        # # plotter.log_model_metrics(result)

        # result = gradient_boosting_with_analysis(
        #     df, target_column, feature_importances, shouldPlot=plotter.shouldPlot
        # )
        # # plotter.log_model_metrics(result)

        # result = neural_network_with_analysis(
        #     df, target_column, feature_importances, shouldPlot=plotter.shouldPlot
        # )
        # # plotter.log_model_metrics(result)

        # result = knn_model_with_analysis(
        #     df, target_column, feature_importances, 5, shouldPlot=plotter.shouldPlot
        # )
        # # plotter.log_model_metrics(result)

        # result = naive_bayes_model_with_analysis(
        #     df, target_column, feature_importances, shouldPlot=plotter.shouldPlot
        # )
        # plotter.log_model_metrics(result)

    # else:
    #     print("Executing in Parallel, and not plotting")

    #     print("Getting feature importances")
    #     feature_importances = get_feature_importance(df, target_column, top_n=5)

    #     for model, data in feature_importances.items():
    #         features = data["Feature"]
    #         print(f"{model}\n = {features}")

    #     def logistic_regression_analysis_wrapper():
    #         return logistic_regression_with_analysis(
    #             df, target_column, feature_importances, shouldPlot=plotter.shouldPlot
    #         )

    #     def random_forest_analysis_wrapper():
    #         return random_forest_with_analysis(
    #             df, target_column, feature_importances, shouldPlot=plotter.shouldPlot
    #         )

    #     def gradient_boosting_analysis_wrapper():
    #         return gradient_boosting_with_analysis(
    #             df, target_column, feature_importances, shouldPlot=plotter.shouldPlot
    #         )

    #     def neural_network_analysis_wrapper():
    #         return neural_network_with_analysis(
    #             df, target_column, feature_importances, shouldPlot=plotter.shouldPlot
    #         )

    #     def knn_analysis_wrapper():
    #         return knn_model_with_analysis(
    #             df, target_column, feature_importances, 5, shouldPlot=plotter.shouldPlot
    #         )

    #     def naive_bayes_analysis_wrapper():
    #         return naive_bayes_model_with_analysis(
    #             df, target_column, feature_importances, shouldPlot=plotter.shouldPlot
    #         )

    #     # Efeature_importancesecute the analysis functions in parallel
    #     print("Executing Various Classical Models")
    #     with ThreadPoolExecutor() as executor:
    #         futures = [
    #             executor.submit(logistic_regression_analysis_wrapper),
    #             executor.submit(random_forest_analysis_wrapper),
    #             executor.submit(gradient_boosting_analysis_wrapper),
    #             executor.submit(neural_network_analysis_wrapper),
    #             executor.submit(knn_analysis_wrapper),
    #             executor.submit(naive_bayes_analysis_wrapper),
    #         ]

    #         # Wait for all tasks to complete and collect results
    #         for future in futures:
    #             result = future.result()
    #             plotter.log_model_metrics(result)
