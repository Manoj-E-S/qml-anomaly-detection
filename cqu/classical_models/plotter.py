import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Any, List
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import os

import matplotlib
matplotlib.use('Agg')

class Plotter:
    def __init__(self, selected_features:List|None=None, shouldPlot:bool=False):
        self.selected_features = selected_features
        self.shouldPlot = shouldPlot
        print(f"Plotter initialized with shouldPlot={shouldPlot}")

    def set_selected_features(self, selected_features:List):
        self.selected_features = selected_features
        print(f"Selected features set to {selected_features}")

    def plot_cm(self, title, cm:np.ndarray):
        if self.shouldPlot:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Non-Fraud', 'Fraud'],
                        yticklabels=['Non-Fraud', 'Fraud'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.savefig(f"./classical_images/{title}_CM.png")
            plt.close()


    def plot_report(self, title, report:dict):
        if self.shouldPlot:
            report_df = pd.DataFrame(report).transpose()
            print(report_df.head(10))
            tick_labels = report_df.index

            plt.figure(figsize=(10, 6))
            report_df[['precision', 'recall', 'f1-score']].plot(kind='bar', rot=0, ax=plt.gca())
            plt.title('Classification Report Metrics')
            plt.xlabel('Classes')
            plt.ylabel('Scores')
            plt.ylim(0, 1)
            plt.xticks(ticks=range(len(tick_labels)), labels=tick_labels, rotation=45)
            plt.legend(title='Metrics', loc='upper left')
            plt.savefig(f"./classical_images/{title}_report.png")
            plt.close()


    def plot_roc_auc(self, title, y_test, y_proba):
        if self.shouldPlot:
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)

            # Plotting the ROC curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', label='ROC Curve (area = {:.2f})'.format(roc_auc))
            plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
            plt.grid()
            plt.savefig(f"./classical_images/{title}_rocauc.png")
            plt.close()

    def visualize_feature_importance(self, model_name, model):
        if self.shouldPlot:
            if hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(8, 6))
                plt.barh(self.selected_features, model.feature_importances_)
                plt.title(f'Feature Importance in {model_name}')
                plt.xlabel('Importance Score')
                plt.ylabel('Features')
                plt.savefig(f"./classical_images/{model_name}_feature_importance.png")
                plt.close()
    
    def print_results_to_stdout(self, title, report, roc_auc, threshold, class_weights=None):
        print(title)
        print("Classification Report:\n", report)
        print("ROC AUC Score:", roc_auc)
        print('threshold:', threshold)
        if class_weights:
            print('class_weights:', class_weights)

    def log_model_metrics(self, results):
        from tabulate import tabulate

        # Prepare the data for tabular logging
        classification_report = results['classification_report']
        roc_auc_score = results['roc_auc_score']
        threshold = results['threshold']
        class_weights = results.get('class_weights')

        # Classification report table
        classification_table = [
            ['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
        ]

        for cls, metrics in classification_report.items():
            if cls not in ['accuracy']: # [..., 'macro avg', 'weighted avg']:
                classification_table.append([
                    cls,
                    metrics['precision'],
                    metrics['recall'],
                    metrics['f1-score'],
                    metrics['support']
                ])

        # Summary table for overall metrics
        if class_weights is None:
            summary_table = [
                ['Metric', 'Value'],
                ['Accuracy', classification_report['accuracy']],
                ['ROC AUC Score', roc_auc_score],
                ['Threshold', threshold],
            ]
        else:
            summary_table = [
                ['Metric', 'Value'],
                ['Accuracy', classification_report['accuracy']],
                ['ROC AUC Score', roc_auc_score],
                ['Threshold', threshold],
                ['Class Weights', class_weights]
            ]

        # Format the tables
        classification_report_str = tabulate(classification_table, headers="firstrow", tablefmt="grid")
        summary_str = tabulate(summary_table, headers="firstrow", tablefmt="grid")

        # Save the tables to a file
        output_dir = "./classical_images/"
        output_file = os.path.join(output_dir, f"{results['model_name']}_evaluation_results.txt")
        os.makedirs(output_dir, exist_ok=True)

        with open(output_file, "w") as f:
            f.write("Classification Report:\n")
            f.write(classification_report_str + "\n\n")
            f.write("Summary Metrics:\n")
            f.write(summary_str)

        print(f"{results['model_name']} results logged to {output_file}")