import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .metrics import ClassifierMetrics

matplotlib.use("Agg")


def plot_cm(title, cm: np.ndarray):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Non-Fraud", "Fraud"],
        yticklabels=["Non-Fraud", "Fraud"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(f"./classical_images/{title}_CM.png")
    plt.close()


def plot_report(title, report: dict):
    report_df = pd.DataFrame(report).transpose()
    print(report_df.head(10))
    tick_labels = report_df.index

    plt.figure(figsize=(10, 6))
    report_df[["precision", "recall", "f1-score"]].plot(kind="bar", rot=0, ax=plt.gca())
    plt.title("Classification Report Metrics")
    plt.xlabel("Classes")
    plt.ylabel("Scores")
    plt.ylim(0, 1)
    plt.xticks(ticks=range(len(tick_labels)), labels=tick_labels, rotation=45)
    plt.legend(title="Metrics", loc="upper left")
    plt.savefig(f"./classical_images/{title}_report.png")
    plt.close()


def plot_roc_auc(title, y_test, y_proba):
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plotting the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label="ROC Curve (area = {:.2f})".format(roc_auc))
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")  # Diagonal line
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"./classical_images/{title}_rocauc.png")
    plt.close()


def visualize_feature_importance(model_name, selected_features, model):
    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(8, 6))
        plt.barh(selected_features, model.feature_importances_)
        plt.title(f"Feature Importance in {model_name}")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.savefig(f"./classical_images/{model_name}_feature_importance.png")
        plt.close()


def print_results_to_stdout(title, report, roc_auc, threshold, class_weights=None):
    print(title)
    print("Classification Report:\n", report)
    print("ROC AUC Score:", roc_auc)
    print("threshold:", threshold)
    if class_weights:
        print("class_weights:", class_weights)
