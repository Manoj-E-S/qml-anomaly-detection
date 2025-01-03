from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, classification_report, roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from cqu.classical_models.plotter import Plotter


def optimize_threshold(y_proba, y_test, step=0.01, useROC=False):
    """
    Find the optimal threshold that maximizes the F1-score for class 1.

    Parameters:
        y_proba (array): Predicted probabilities for the positive class.
        y_test (array): Ground truth labels.
        step (float): Increment for threshold adjustment.

    Returns:
        dict: Dictionary containing the optimal threshold, corresponding F1-score, and the classification report.
    """

    if useROC:
        threshold = None
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        gmeans = (tpr * (1 - fpr)) ** 0.5
        optimal_idx = gmeans.argmax()
        threshold = 1 - thresholds[optimal_idx]
        return threshold

    best_threshold = 0.0
    best_f1_score = 0.0
    best_report = None

    # Iterate through thresholds from 0.0 to 1.0 with the given step size
    for threshold in [x * step for x in range(int(1 / step) + 1)]:
        # Apply the threshold and classify
        y_pred = (y_proba >= threshold).astype(int)

        # Generate evaluation metrics
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        f1 = report["1"]["f1-score"]  # Get F1-score for class 1

        # Update best threshold if this F1 is better
        if f1 > best_f1_score:
            best_f1_score = f1
            best_threshold = threshold
            best_report = report
            

    return best_threshold, best_report



def logistic_regression_with_analysis(data, target_column, important_features, threshold=None, shouldPlot=False):
    """
    Builds a logistic regression model using ROC curve analysis to determine the best threshold
    and class weights. Evaluates the model's performance on specified important features.

    Parameters:
    - data (pd.DataFrame): The input dataset.
    - target_column (str): The name of the target variable column.
    - important_features (pd.DataFrame): A DataFrame with feature names and their importance.
    - threshold (float): A predefined threshold. If None, optimal threshold will be determined using the ROC curve.

    Returns:
    - dict: A dictionary containing precision, recall, F1-score, ROC AUC score, and the classification report.
    """
    plotter = Plotter(shouldPlot=shouldPlot)

    feature_names = important_features['logistic_regression']['Feature'].tolist()
    # feature_names = ['Amount', 'V3', 'V14', 'V17', 'V9']
    plotter.set_selected_features(feature_names)

    # Separate features and target
    X = data[feature_names]
    y = data[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Compute class weights
    class_weights = {0: len(y) / (2 * (y == 0).sum()), 1: len(y) / (2 * (y == 1).sum())} #change class weights formulae
    class_weights = {0: 1, 1: 35}

    # Fit Logistic Regression model
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weights)
    model.fit(X_train, y_train)

    # Predict probabilities for the positive class
    y_proba = model.predict_proba(X_test)[:, 1]

    # Determine optimal threshold using ROC curve analysis if threshold not provided
    if threshold is None:
        threshold = optimize_threshold(y_proba, y_test, step=0.01, useROC=True)

    # Apply the threshold to classify
    y_pred = (y_proba >= threshold).astype(int)

    # Generate evaluation metrics
    dict_report = classification_report(y_test, y_pred, output_dict=True)
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Print metrics
    plotter.print_results_to_stdout("\n\nLogistic Regression", report, roc_auc, threshold, class_weights)
    plotter.visualize_feature_importance('logistic_regression', model)
    plotter.plot_cm("LogisticRegression", confusion_matrix(y_test, y_pred))
    plotter.plot_report("LogisticRegression", dict_report)
    plotter.plot_roc_auc("LogisticRegression", y_test, y_proba)
    
    results = {
        'model_name': 'LogisticRegression',
        'classification_report': dict_report,
        'roc_auc_score': roc_auc,
        'threshold': threshold,
        'class_weights': class_weights
    }
    return results


def random_forest_with_analysis(data, target_column, important_features, threshold=None, shouldPlot=False):
    """
    Builds a Random Forest model, optimizes the decision threshold using predict_proba, and 
    computes the most optimal class weights based on class imbalance.
    
    Parameters:
    - data (pd.DataFrame): The input dataset.
    - target_column (str): The name of the target variable column.
    - important_features (pd.DataFrame): A DataFrame with feature names and their importance.
    - threshold (float): A predefined threshold. If None, optimal threshold will be determined using the ROC curve.

    Returns:
    - dict: A dictionary containing precision, recall, F1-score, ROC AUC score, and the classification report.
    """
    plotter = Plotter(shouldPlot=shouldPlot)

    # Use only the important features
    feature_names = important_features['random_forest']['Feature'].tolist()
    # feature_names = ['V17', 'V12', 'V14', 'V10', 'V16']
    plotter.set_selected_features(feature_names)

    X = data[feature_names]
    y = data[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Compute optimal class weights
    # class_weights = {0: len(y) / (2 * (y == 0).sum()), 1: len(y) / (2 * (y == 1).sum())}
    class_weights = {0: 1, 1: 35}

    # Fit Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights)
    model.fit(X_train, y_train)

    # Predict probabilities for the positive class
    y_proba = model.predict_proba(X_test)[:, 1]
    threshold, _ = optimize_threshold(y_proba, y_test, 0.01)

    # Apply the threshold to classify
    y_pred = (y_proba >= threshold).astype(int)

    # Generate evaluation metrics
    dict_report = classification_report(y_test, y_pred, output_dict=True)
    report =  classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Print metrics
    plotter.print_results_to_stdout("\n\nRandom Forest", report, roc_auc, threshold, class_weights)
    plotter.visualize_feature_importance('random_forest', model)
    plotter.plot_cm("RandomForest", confusion_matrix(y_test, y_pred))
    plotter.plot_report("RandomForest", dict_report)
    plotter.plot_roc_auc("RandomForest", y_test, y_proba)

    results = {
        'model_name': 'RandomForest',
        'classification_report': dict_report,
        'roc_auc_score': roc_auc,
        'threshold': threshold,
        'class_weights': class_weights
    }

    return results

def gradient_boosting_with_analysis(data, target_column, important_features, threshold=None, shouldPlot=False):
    """
    Builds a Gradient Boosting model (XGBoost), optimizes the decision threshold using predict_proba, 
    and computes the most optimal scale_pos_weight based on class imbalance.
    
    Parameters:
    - data (pd.DataFrame): The input dataset.
    - target_column (str): The name of the target variable column.
    - important_features (pd.DataFrame): A DataFrame with feature names and their importance.
    - threshold (float): A predefined threshold. If None, optimal threshold will be determined using the ROC curve.

    Returns:
    - dict: A dictionary containing precision, recall, F1-score, ROC AUC score, and the classification report.
    """
    plotter = Plotter(shouldPlot=shouldPlot)

    # Use only the important features
    feature_names = important_features['gradient_boosting']['Feature'].tolist()
    # feature_names = ['V14', 'V27', 'V10', 'V17', 'V11']
    plotter.set_selected_features(feature_names)

    X = data[feature_names]
    y = data[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Compute optimal scale_pos_weight
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    # Fit Gradient Boosting (XGBoost) model
    model = XGBClassifier(
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        # use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    # Predict probabilities for the positive class
    y_proba = model.predict_proba(X_test)[:, 1]

    # Apply threshold and get report
    threshold, dict_report = optimize_threshold(y_proba, y_test, 0.01)
    y_pred = (y_proba >= threshold).astype(int)

    # Generate evaluation metrics
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Print metrics
    plotter.print_results_to_stdout("\n\nGradient Boosting", report, roc_auc, threshold)
    plotter.visualize_feature_importance('gradient_boosting', model)
    plotter.plot_cm("GradientBoosting", confusion_matrix(y_test, y_pred))
    plotter.plot_report("GradientBoosting", dict_report)
    plotter.plot_roc_auc("GradientBoosting", y_test, y_proba)

    results = {
        'model_name': 'GradientBoosting',
        'classification_report': dict_report,
        'roc_auc_score': roc_auc,
        'threshold': threshold,
        'scale_pos_weight': scale_pos_weight
    }

    return results


def neural_network_with_analysis(data, target_column, important_features, threshold=None, shouldPlot=False):
    """
    Builds a Neural Network model, adjusts the class weights during training, optimizes the decision threshold,
    and computes evaluation metrics.

    Parameters:
    - data (pd.DataFrame): The input dataset.
    - target_column (str): The name of the target variable column.
    - important_features (pd.DataFrame): A DataFrame with feature names and their importance.
    - threshold (float): A predefined threshold. If None, optimal threshold will be determined using the ROC curve.

    Returns:
    - dict: A dictionary containing precision, recall, F1-score, ROC AUC score, and the classification report.
    """

    class FraudDetectionNN(nn.Module):
        def __init__(self, input_size):
            super(FraudDetectionNN, self).__init__()

            # Layers of the neural network
            self.layers = nn.Sequential(
                nn.Linear(input_size, 64),           # Input Layer
                nn.ReLU(),                           # Activation
                nn.BatchNorm1d(64),                  # Batch Normalization
                nn.Dropout(0.3),                     # Dropout Layer to prevent overfitting

                nn.Linear(64, 32),                   # Hidden Layer 1
                nn.ReLU(),                           # Activation
                nn.BatchNorm1d(32),                  # Batch Normalization
                nn.Dropout(0.2),                     # Dropout Layer

                nn.Linear(32, 16),                   # Hidden Layer 2
                nn.ReLU(),                           # Activation
                nn.BatchNorm1d(16),                  # Batch Normalization

                nn.Linear(16, 1),                    # Output Layer
                nn.Sigmoid()                         # Sigmoid activation for binary classification
            )

        def forward(self, x):
            return self.layers(x)
    

    plotter = Plotter(shouldPlot=shouldPlot)

    # Use only the important features
    feature_names = important_features['neural_network']['Feature'].tolist()
    # feature_names = ['V17', 'V14', 'V12', 'V16', 'V10']
    plotter.set_selected_features(feature_names)
    
    X = data[feature_names]
    y = data[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1) # 1d tensor to 2d tensor
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1) # 1d tensor to 2d tensor

    # Compute class weights
    class_counts = np.bincount(y_train)
    class_weights = {i: max(class_counts) / count for i, count in enumerate(class_counts)}

    # Build the neural network model
    model = FraudDetectionNN(X_train.shape[1])

    # Loss function and optimizer
    lossfn = nn.BCELoss()  # Binary Cross-Entropy Loss for classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train_tensor)
        loss = lossfn(outputs, y_train_tensor)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    # Evaluation of the model
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor) # Predicted probabilities
        threshold, _ = optimize_threshold(test_outputs.cpu().numpy().flatten(), y_test, 0.01)
        test_predictions_array = (test_outputs.cpu().numpy().flatten() >= threshold).astype(int) # Convert probabilities to binary predictions

        # Convert tensors to numpy arrays
        y_test_array = y_test_tensor.cpu().numpy().flatten()
        test_proba_array = test_outputs.cpu().numpy().flatten()

        cm = confusion_matrix(
            y_test_array, 
            test_predictions_array
        )

        dict_report = classification_report(
            y_test_array, 
            test_predictions_array, 
            output_dict=True
        )

        report = classification_report(
            y_test_array, 
            test_predictions_array
        )

        accuracy = (test_predictions_array == y_test).sum().item() / len(y_test)
        print(f'Accuracy on test data: {accuracy:.4f}')
        roc_auc = roc_auc_score(y_test_array, test_proba_array)

        # Print metrics
        plotter.print_results_to_stdout("\n\nNeural Network", report, roc_auc, threshold)
        plotter.visualize_feature_importance('neural_network', model)
        plotter.plot_cm("NeuralNetwork", cm)
        plotter.plot_report("NeuralNetwork", dict_report)
        plotter.plot_roc_auc("NeuralNetwork", y_test_array, test_predictions_array)

        results = {
            'model_name': 'NeuralNetwork',
            'classification_report': dict_report,
            'roc_auc_score': roc_auc,
            'threshold': threshold,
            'class_weights': class_weights
        }

    return results


def knn_model_with_analysis(data, target_column, important_features, n_neighbors=5, threshold=None, shouldPlot=False):
    """
    Builds a K-Nearest Neighbors (KNN) model, optimizes the threshold using predict_proba,
    and evaluates model performance.

    Parameters:
    - data (pd.DataFrame): The input dataset.
    - target_column (str): The name of the target variable column.
    - important_features (pd.DataFrame): A DataFrame with feature names and their importance.
    - n_neighbors (int): Number of neighbors to use for KNN.
    - threshold (float): A predefined threshold. If None, optimal threshold will be determined using ROC curve.

    Returns:
    - dict: A dictionary containing precision, recall, F1-score, ROC AUC score, and the classification report.
    """
    plotter = Plotter(shouldPlot=shouldPlot)

    # Use only the important features
    feature_names = important_features['knn']['Feature'].tolist()
    # feature_names = ['V17', 'V14', 'V12', 'V10', 'V16']
    plotter.set_selected_features(feature_names)

    X = data[feature_names].values
    y = data[target_column].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize and train the KNN model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    knn.fit(X_train, y_train)

    # Predict probabilities for the positive class
    y_proba = knn.predict_proba(X_test)[:, 1]

    # Apply the threshold to classify
    threshold, _ = optimize_threshold(y_proba, y_test, 0.01)
    y_pred = (y_proba >= threshold).astype(int)

    # Generate evaluation metrics
    dict_report = classification_report(y_test, y_pred, output_dict=True)
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Print metrics
    plotter.print_results_to_stdout("\n\nKnn Model", report, roc_auc, threshold)
    plotter.plot_cm("KNN", confusion_matrix(y_test, y_pred))
    plotter.plot_report("KNN", dict_report)
    plotter.plot_roc_auc("KNN", y_test, y_proba)

    results = {
        'model_name': 'KNN',
        'classification_report': dict_report,
        'roc_auc_score': roc_auc,
        'threshold': threshold,
    }

    return results


def naive_bayes_model_with_analysis(data, target_column, important_features, threshold=None, shouldPlot=False):
    """
    Builds a Naive Bayes model, optimizes the threshold using predict_proba,
    and evaluates model performance.

    Parameters:
    - data (pd.DataFrame): The input dataset.
    - target_column (str): The name of the target variable column.
    - important_features (pd.DataFrame): A DataFrame with feature names and their importance.
    - threshold (float): A predefined threshold. If None, optimal threshold will be determined using ROC curve.

    Returns:
    - dict: A dictionary containing precision, recall, F1-score, ROC AUC score, and the classification report.
    """
    plotter = Plotter(shouldPlot=shouldPlot)

    # Use only the important features
    feature_names = important_features['naive_bayes']['Feature'].tolist()
    # feature_names = ['Time', 'Amount', 'V14', 'V3', 'V17']
    plotter.set_selected_features(feature_names)

    X = data[feature_names].values
    y = data[target_column].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize and train the Naive Bayes model
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train, y_train)

    # Predict probabilities for the positive class
    y_proba = naive_bayes.predict_proba(X_test)[:, 1]

    # Apply the threshold to classify
    threshold, _ = optimize_threshold(y_proba, y_test, 0.01)
    y_pred = (y_proba >= threshold).astype(int)

    # Generate evaluation metrics
    dict_report = classification_report(y_test, y_pred, output_dict=True)
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Print metrics
    plotter.print_results_to_stdout("\n\nNaive Bayes Model", report, roc_auc, threshold)
    plotter.visualize_feature_importance('naive_bayes', naive_bayes)
    plotter.plot_cm("NaiveBayes", confusion_matrix(y_test, y_pred))
    plotter.plot_report("NaiveBayes", dict_report)
    plotter.plot_roc_auc("NaiveBayes", y_test, y_proba)

    results = {
        'model_name': 'NaiveBayes',
        'classification_report': dict_report,
        'roc_auc_score': roc_auc,
        'threshold': threshold,
    }

    return results
