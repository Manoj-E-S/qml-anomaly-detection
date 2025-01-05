"""
'ClassifierMetrics' dataclass is used to store the metrics of a classifier. along with the helper
function 'get_metrics' to calculate the metrics of a classifier given the true and predicted
values of the classifier.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from tabulate import tabulate

from cqu.typing import ModelType


@dataclass
class ClassifierMetrics:
    model_type: ModelType
    feature_importances: pd.DataFrame | None
    report: Dict
    accuracy: float
    confusion_matrix: np.ndarray
    roc_curve: tuple[np.ndarray, np.ndarray, np.ndarray]
    roc_auc: float
    class_weights: Dict[int | str, float] | None

    def to_string(self) -> str:
        classification_table_str = self.__get_classifcation_table_str()
        summary_table_str = self.__get_summary_table_str()
        confusion_matrix_str = self.__get_confusion_matrix_str()

        return (
            f"Model Type: {self.model_type}\n\n"
            f"Classification Report:\n{classification_table_str}\n\n"
            f"Summary:\n{summary_table_str}\n\n"
            f"Confusion Matrix:\n{confusion_matrix_str}"
        )

    def __get_classifcation_table_str(self) -> str:
        classification_table = [
            ["Class", "Precision", "Recall", "F1-Score", "Support"],
        ]

        for cls, metrics in self.report.items():
            if cls not in ["accuracy"]:
                classification_table.append(
                    [
                        cls,
                        metrics["precision"],
                        metrics["recall"],
                        metrics["f1-score"],
                        metrics["support"],
                    ]
                )

        classification_table_str = tabulate(
            classification_table, headers="firstrow", tablefmt="grid"
        )

        return classification_table_str

    def __get_summary_table_str(self) -> str:
        summary_table = [
            ["Metric", "Value"],
            ["Accuracy", self.accuracy],
            ["ROC AUC Score", self.roc_auc],
            ["Threshold", self.roc_curve[2][1]],
        ]

        if self.class_weights is not None:
            summary_table.append(["Class Weights", self.class_weights])

        summary_table_str = tabulate(summary_table, headers="firstrow", tablefmt="grid")

        return summary_table_str

    def __get_confusion_matrix_str(self) -> str:
        return tabulate(self.confusion_matrix, tablefmt="grid")

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self):
        return self.to_string()


def get_metrics(
    model_type: ModelType,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    feature_importances: pd.DataFrame = None,
    class_wieghts: dict = None,
) -> ClassifierMetrics:
    report = classification_report(y_true, y_pred, output_dict=True)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_c = (fpr, tpr, thresholds)
    roc_auc = auc(fpr, tpr)

    return ClassifierMetrics(
        model_type=model_type,
        feature_importances=feature_importances,
        report=report,
        accuracy=accuracy,
        confusion_matrix=cm,
        roc_curve=roc_c,
        roc_auc=roc_auc,
        class_weights=class_wieghts,
    )
