from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def evaluate_predictions(y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict:
    preds = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        preds,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )
    cm = confusion_matrix(y_true, preds, labels=np.arange(len(class_names)))
    return {
        "accuracy": accuracy_score(y_true, preds),
        "macro_f1": float(np.mean(f1)),
        "confusion_matrix": cm.astype(int).tolist(),
        "per_class": [
            {
                "class_name": class_name,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx, class_name in enumerate(class_names)
        ],
    }


def save_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_results_csv(results: dict[str, dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["setting", "accuracy", "macro_f1"])
        writer.writeheader()
        for setting, metrics in results.items():
            writer.writerow(
                {
                    "setting": setting,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                }
            )


def plot_training_curves(history: dict[str, list[float]], path: str | Path, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train loss")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="val loss")
    if "hard_loss" in history:
        axes[0].plot(epochs, history["hard_loss"], label="hard term")
    if "soft_loss" in history:
        axes[0].plot(epochs, history["soft_loss"], label="soft term")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train acc")
    axes[1].plot(epochs, history["val_acc"], label="val acc")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_uncertainty_histograms(score_dict: dict[str, np.ndarray], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, (name, values) in zip(axes, score_dict.items()):
        axis.hist(values, bins=30, alpha=0.85)
        axis.set_title(name.replace("_", " ").title())
        axis.set_xlabel("Score")
        axis.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_projection(points_2d: np.ndarray, selected_mask: np.ndarray, path: str | Path, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(points_2d[~selected_mask, 0], points_2d[~selected_mask, 1], s=8, alpha=0.25, label="unlabeled pool")
    ax.scatter(points_2d[selected_mask, 0], points_2d[selected_mask, 1], s=18, alpha=0.9, label="selected coreset")
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], path: str | Path, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, int(cm[row, col]), ha="center", va="center", fontsize=7)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_accuracy_bar(results: dict[str, dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(results.keys())
    accuracies = [results[label]["accuracy"] for label in labels]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, accuracies)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Teacher and Student Comparison")
    ax.set_ylim(0.0, 1.0)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
