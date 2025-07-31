#!/usr/bin/env python3
"""
Iris Classifier - Main Training Script

This script loads the Iris dataset, trains a Logistic Regression model with
hyperparameter tuning, and generates evaluation metrics and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from typing import Tuple, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def load_data() -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load the Iris dataset and return as DataFrame and target array.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: Features DataFrame and target array
    """
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df["species"] = [iris.target_names[i] for i in iris.target]

    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['species'].value_counts()}")

    return df, iris.target


def create_pipeline() -> Pipeline:
    """
    Create a scikit-learn pipeline with StandardScaler and LogisticRegression.

    Returns:
        Pipeline: The configured pipeline
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logisticregression", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    return pipeline


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Pipeline, dict]:
    """
    Train the model using GridSearchCV for hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training targets

    Returns:
        Tuple[Pipeline, dict]: Best model and CV results
    """
    pipeline = create_pipeline()

    # Define parameter grid for GridSearchCV
    param_grid = {"logisticregression__C": [0.1, 1, 10]}

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
    )

    print("Training model with GridSearchCV...")
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.cv_results_


def evaluate_model(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Evaluate the model and print classification report.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["setosa", "versicolor", "virginica"]
        )
    )

    return y_pred


def plot_confusion_matrix(
    y_test: np.ndarray, y_pred: np.ndarray, save_path: str = "cmatrix.png"
) -> None:
    """
    Create and save confusion matrix plot.

    Args:
        y_test: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["setosa", "versicolor", "virginica"],
        yticklabels=["setosa", "versicolor", "virginica"],
    )

    plt.title("Confusion Matrix - Iris Classifier", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_validation_curve(cv_results: dict) -> None:
    """
    Plot validation curve showing CV scores for different C values.

    Args:
        cv_results: Results from GridSearchCV
    """
    C_values = [0.1, 1, 10]
    mean_scores = cv_results["mean_test_score"]
    std_scores = cv_results["std_test_score"]

    plt.figure(figsize=(10, 6))
    plt.errorbar(C_values, mean_scores, yerr=std_scores, marker="o", capsize=5)
    plt.xscale("log")
    plt.xlabel("C (Regularization Parameter)", fontsize=12)
    plt.ylabel("Cross-Validation Accuracy", fontsize=12)
    plt.title("Validation Curve - Logistic Regression", fontsize=16, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig("validation_curve.png", dpi=300, bbox_inches="tight")
    print("Validation curve saved to validation_curve.png")
    plt.close()


def main() -> None:
    """
    Main function to orchestrate the entire training pipeline.
    """
    print("=== Iris Classifier Training Pipeline ===\n")

    # Load data
    df, target = load_data()

    # Prepare features and target
    feature_names = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    X = df[feature_names].values
    y = target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Train model
    best_model, cv_results = train_model(X_train, y_train)

    # Evaluate model
    y_pred = evaluate_model(best_model, X_test, y_test)

    # Generate plots
    plot_confusion_matrix(y_test, y_pred)
    plot_validation_curve(cv_results)

    # Save model
    joblib.dump(best_model, "model.joblib")
    print("\nModel saved to model.joblib")

    print("\n=== Training Complete! ===")


if __name__ == "__main__":
    main()
