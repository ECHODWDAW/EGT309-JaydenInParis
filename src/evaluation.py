
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)


class ModelEvaluator:
    """Shared evaluation pipeline for classification models."""

    def __init__(self, model_name="Model"):
        self.model_name = model_name
        self.results = {}

    def evaluate(self, y_true, y_pred, y_prob=None):
        """Calculate classification performance metrics."""

        self.results = {
            "model": self.model_name,

            "accuracy": accuracy_score(
                y_true, y_pred
            ),

            "precision": precision_score(
                y_true, y_pred, zero_division=0
            ),

            "recall": recall_score(
                y_true, y_pred, zero_division=0
            ),

            "f1_score": f1_score(
                y_true, y_pred, zero_division=0
            ),

            "roc_auc": np.nan,
            "average_precision": np.nan
        }

        if y_prob is not None:
            if len(np.unique(y_true)) == 2:

                self.results["roc_auc"] = roc_auc_score(
                    y_true, y_prob
                )

                self.results["average_precision"] = (
                    average_precision_score(y_true, y_prob)
                )

        return self.results.copy()

    def print_results(self):
        """Display the calculated evaluation metrics."""

        if not self.results:
            raise ValueError("Evaluate a model first.")

        print(f"\n--- {self.model_name} Evaluation ---")

        for metric, value in self.results.items():

            if metric != "model":
                print(f"{metric}: {value:.4f}")

    def print_classification_report(self, y_true, y_pred):
        """Display performance for each target class."""

        print("\n--- Classification Report ---")

        print(
            classification_report(
                y_true,
                y_pred,
                labels=[0, 1],
                target_names=[
                    "No Maintenance",
                    "Maintenance Required"
                ],
                zero_division=0
            )
        )

    def print_confusion_matrix(self, y_true, y_pred):
        """Display the four confusion matrix outcomes."""

        matrix = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        )

        tn, fp, fn, tp = matrix.ravel()

        print("\n--- Confusion Matrix ---")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives: {tp}")

        return matrix

    def plot_confusion_matrix(self, y_true, y_pred):
        """Generate a confusion matrix visualisation."""

        fig, ax = plt.subplots(figsize=(6, 5))

        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            labels=[0, 1],
            display_labels=[
                "No Maintenance",
                "Maintenance"
            ],
            cmap="Blues",
            ax=ax
        )

        ax.set_title(
            f"{self.model_name} - Confusion Matrix"
        )

        fig.tight_layout()

        return fig

    def plot_roc_curve(self, y_true, y_prob):
        """Generate a ROC curve visualisation."""

        fig, ax = plt.subplots(figsize=(6, 5))

        RocCurveDisplay.from_predictions(
            y_true,
            y_prob,
            name=self.model_name,
            ax=ax
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="grey"
        )

        ax.set_title(
            f"{self.model_name} - ROC Curve"
        )

        fig.tight_layout()

        return fig

    def plot_precision_recall_curve(self, y_true, y_prob):
        """Generate a precision-recall curve."""

        fig, ax = plt.subplots(figsize=(6, 5))

        PrecisionRecallDisplay.from_predictions(
            y_true,
            y_prob,
            name=self.model_name,
            ax=ax
        )

        baseline = np.mean(np.asarray(y_true) == 1)

        ax.axhline(
            baseline,
            linestyle="--",
            color="grey",
            label="Baseline"
        )

        ax.legend()

        ax.set_title(
            f"{self.model_name} - Precision-Recall Curve"
        )

        fig.tight_layout()

        return fig

    def save_results(self, output_path):
        """Save evaluation metrics to a CSV file."""

        if not self.results:
            raise ValueError("Evaluate a model first.")

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        pd.DataFrame([self.results]).to_csv(
            output_path,
            index=False
        )

        print(f"Results saved to: {output_path}")

    def run(self, y_true, y_pred, y_prob=None):
        """Execute the shared evaluation pipeline."""

        results = self.evaluate(
            y_true, y_pred, y_prob
        )

        self.print_results()

        self.print_classification_report(
            y_true, y_pred
        )

        self.print_confusion_matrix(
            y_true, y_pred
        )

        print("\nModel evaluation completed successfully.")

        return results


if __name__ == "__main__":

    # Sample data for testing the evaluation pipeline.
    y_true = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

    y_pred = [0, 0, 0, 0, 0, 1, 1, 1, 0, 1]

    y_prob = [
        0.1, 0.2, 0.1, 0.3, 0.2,
        0.7, 0.8, 0.9, 0.4, 0.8
    ]

    evaluator = ModelEvaluator("Test Model")

    evaluator.run(y_true, y_pred, y_prob)
