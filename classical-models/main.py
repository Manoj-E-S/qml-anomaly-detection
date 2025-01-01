from .important_features import get_feature_importance
from .models import logistic_regression_with_analysis, random_forest_with_analysis, gradient_boosting_with_analysis, neural_network_with_analysis, knn_model_with_analysis, naive_bayes_model_with_analysis
import pandas as pd

if __name__ == '__main__':
    
    data = pd.read_csv("datasets\ccfraud\creditcard.csv")
    target_column = "Class"  # Replace with your target column name

    x = get_feature_importance(data, target_column, top_n=5)
    for model, data in x.items():
        features = data['Feature']
        print(f"{model}\n = {features}")

    logistic_regression_with_analysis(data, target_column, x)
    random_forest_with_analysis(data, target_column, x)
    gradient_boosting_with_analysis(data, target_column, x)
    neural_network_with_analysis(data, target_column, x)
    knn_model_with_analysis(data, target_column, x, 5)
    naive_bayes_model_with_analysis(data, target_column, x)