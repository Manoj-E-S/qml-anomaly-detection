from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, classification_report, roc_auc_score
import pandas as pd

def logistic_regression_with_analysis(data, target_column, important_features, threshold=None):
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
    # Use only the important features
    feature_names = important_features['logistic_regression']['Feature'].tolist()
    X = data[feature_names]
    y = data[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Compute class weights
    class_weights = {0: len(y) / (2 * (y == 0).sum()), 1: len(y) / (2 * (y == 1).sum())}

    # Fit Logistic Regression model
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weights)
    model.fit(X_train, y_train)

    # Predict probabilities for the positive class
    y_proba = model.predict_proba(X_test)[:, 1]

    # Determine optimal threshold using ROC curve analysis if threshold not provided
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        gmeans = (tpr * (1 - fpr)) ** 0.5
        optimal_idx = gmeans.argmax()
        threshold = thresholds[optimal_idx]

    # Apply the threshold to classify
    y_pred = (y_proba >= threshold).astype(int)

    # Generate evaluation metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Print and return results
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc)

    results = {
        'classification_report': report,
        'roc_auc_score': roc_auc,
        'threshold': threshold,
        'class_weights': class_weights
    }
    print(results)

    return results
