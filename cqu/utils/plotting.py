import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import PLOT_FOLDER_NAME
from .metrics import ClassifierMetrics


def agg_plot(func):
    def wrapper(*args, **kwargs):
        original_backend = matplotlib.get_backend()
        matplotlib.use("Agg", force=True)
        try:
            func(*args, **kwargs)
        finally:
            matplotlib.use(original_backend, force=True)

    return wrapper


def plot_all_metrics(model_name: str, metrics: ClassifierMetrics) -> None:
    plot_confusion_matrix(model_name, metrics.confusion_matrix)
    plot_report(model_name, metrics.report)
    plot_roc_auc(model_name, metrics.roc_curve, metrics.roc_auc)
    plot_feature_importance(model_name, metrics.feature_importances)


@agg_plot
def plot_confusion_matrix(
    model_name: str, confusion_matrix: np.ndarray, class_labels: List[str] = ["0", "1"]
) -> None:
    if len(class_labels) != 2:
        raise ValueError("Class labels must be a list of length 2")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(os.path.join(PLOT_FOLDER_NAME, f"{model_name}_CM.png"))
    plt.close()


@agg_plot
def plot_report(model_name: str, classification_report: dict) -> None:
    report_df = pd.DataFrame(classification_report).transpose()
    tick_labels = report_df.index

    plt.figure(figsize=(10, 6))
    report_df[["precision", "recall", "f1-score"]].plot(kind="bar", rot=0, ax=plt.gca())
    plt.title("Classification Report Metrics")
    plt.xlabel("Classes")
    plt.ylabel("Scores")
    plt.ylim(0, 1)
    plt.xticks(ticks=range(len(tick_labels)), labels=tick_labels, rotation=45)
    plt.legend(title="Metrics", loc="upper left")
    plt.savefig(os.path.join(PLOT_FOLDER_NAME, f"{model_name}_report.png"))
    plt.close()


@agg_plot
def plot_roc_auc(
    model_name: str,
    roc_curve: tuple[np.ndarray, np.ndarray, np.ndarray],
    roc_auc: float,
) -> None:
    fpr, tpr, _ = roc_curve

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label="ROC Curve (area = {:.2f})".format(roc_auc))
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")  # Diagonal line
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(os.path.join(PLOT_FOLDER_NAME, f"{model_name}_rocauc.png"))
    plt.close()


@agg_plot
def plot_feature_importance(
    model_name: str, feature_importances: pd.DataFrame | None
) -> None:
    if feature_importances is None:
        return

    plt.figure(figsize=(8, 6))
    plt.barh(feature_importances["feature"], feature_importances["importance"])
    plt.title(f"Feature Importance in {model_name}")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.savefig(os.path.join(PLOT_FOLDER_NAME, f"{model_name}_feature_importance.png"))
    plt.close()
