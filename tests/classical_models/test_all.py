from concurrent.futures import ThreadPoolExecutor
from cqu.classical_models.important_features import get_feature_importance
from cqu.classical_models.models import (
    logistic_regression_with_analysis, 
    random_forest_with_analysis, 
    gradient_boosting_with_analysis, 
    knn_model_with_analysis, 
    naive_bayes_model_with_analysis, 
    neural_network_with_analysis
)
from cqu.classical_models.plotter import Plotter
from . import (
    get_dataset, 
    reduce_dataset
)

def all_models_helper(df, target_column, parallel=True):
    
    print("Getting feature importances")
    feature_importances = get_feature_importance(df, target_column, top_n=5)

    for model, data in feature_importances.items():
        features = data['Feature']
        print(f"{model}\n = {features}")

    if parallel == False:
        print("Executing Sequentially, and plotting")
        plotter = Plotter(shouldPlot=True)

        print("Getting feature importances")
        feature_importances = get_feature_importance(df, target_column, top_n=5)

        for model, data in feature_importances.items():
            features = data['Feature']
            print(f"{model}\n = {features}")

        result = logistic_regression_with_analysis(df, target_column, feature_importances, shouldPlot=plotter.shouldPlot)
        plotter.log_model_metrics(result)

        result = random_forest_with_analysis(df, target_column, feature_importances, shouldPlot=plotter.shouldPlot)
        plotter.log_model_metrics(result)

        result = gradient_boosting_with_analysis(df, target_column, feature_importances, shouldPlot=plotter.shouldPlot)
        plotter.log_model_metrics(result)

        result = neural_network_with_analysis(df, target_column, feature_importances, shouldPlot=plotter.shouldPlot)
        plotter.log_model_metrics(result)

        result = knn_model_with_analysis(df, target_column, feature_importances, 5, shouldPlot=plotter.shouldPlot)
        plotter.log_model_metrics(result)

        result = naive_bayes_model_with_analysis(df, target_column, feature_importances, shouldPlot=plotter.shouldPlot)
        plotter.log_model_metrics(result)

        return

    print("Executing in Parallel, and not plotting")
    plotter = Plotter(shouldPlot=False)

    print("Getting feature importances")
    feature_importances = get_feature_importance(df, target_column, top_n=5)

    for model, data in feature_importances.items():
        features = data['Feature']
        print(f"{model}\n = {features}")


    def logistic_regression_analysis_wrapper():
        return logistic_regression_with_analysis(df, target_column, feature_importances, shouldPlot=plotter.shouldPlot)

    def random_forest_analysis_wrapper():
        return random_forest_with_analysis(df, target_column, feature_importances, shouldPlot=plotter.shouldPlot)

    def gradient_boosting_analysis_wrapper():
        return gradient_boosting_with_analysis(df, target_column, feature_importances, shouldPlot=plotter.shouldPlot)

    def neural_network_analysis_wrapper():
        return neural_network_with_analysis(df, target_column, feature_importances, shouldPlot=plotter.shouldPlot)

    def knn_analysis_wrapper():
        return knn_model_with_analysis(df, target_column, feature_importances, 5, shouldPlot=plotter.shouldPlot)

    def naive_bayes_analysis_wrapper():
        return naive_bayes_model_with_analysis(df, target_column, feature_importances, shouldPlot=plotter.shouldPlot)

    # Efeature_importancesecute the analysis functions in parallel
    print("Executing Various Classical Models")
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(logistic_regression_analysis_wrapper),
            executor.submit(random_forest_analysis_wrapper),
            executor.submit(gradient_boosting_analysis_wrapper),
            executor.submit(neural_network_analysis_wrapper),
            executor.submit(knn_analysis_wrapper),
            executor.submit(naive_bayes_analysis_wrapper),
        ]
        
        # Wait for all tasks to complete and collect results
        for future in futures:
            result = future.result()
            plotter.log_model_metrics(result)


def test_all_models():
    print("Getting fraud dataset")
    df = get_dataset("../datasets/ccfraud/creditcard.csv")
    target_column = "class"

    all_models_helper(df, target_column)


def test_all_models_with_reduced_data():
    print("Getting fraud dataset")
    ds = get_dataset("../datasets/ccfraud/creditcard.csv")
    ds = reduce_dataset(ds, 100, 10)
    target_column = "class"

    all_models_helper(ds, target_column)