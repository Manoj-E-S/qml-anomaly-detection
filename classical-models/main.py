from .important_features import get_feature_importance
from .models import logistic_regression_with_analysis, random_forest_with_analysis, gradient_boosting_with_analysis, neural_network_with_analysis, knn_model_with_analysis, naive_bayes_model_with_analysis
import pandas as pd

if __name__ == '__main__':
    
    data = pd.read_csv("datasets\ccfraud\creditcard.csv")
    target_column = "Class"  # Replace with your target column name

    # x = get_feature_importance(data, target_column, top_n=5)
    # print(x)
    # print(x['logistic_regression']['Feature'].tolist())
    # print(x['random_forest']['Feature'].tolist())

    # logistic_regression_with_analysis(data, target_column, x)
    # random_forest_with_analysis(data, target_column, 1)
    # gradient_boosting_with_analysis(data, target_column, 1)
    # neural_network_with_analysis(data, target_column, 1)
    # knn_model_with_analysis(data, target_column, 1, 5)
    naive_bayes_model_with_analysis(data, target_column, 1)