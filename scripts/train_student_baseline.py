from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import get_student_splits, load_semisupervised_fashion_mnist, set_seed
from src.metrics import evaluate_predictions, plot_training_curves, save_json
from src.student_manual import ManualSoftmaxRegression


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the manual softmax-regression baseline student.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    if args.epochs is not None:
        config["student"]["epochs"] = args.epochs
    if args.learning_rate is not None:
        config["student"]["learning_rate"] = args.learning_rate

    set_seed(int(config["seed"]))
    bundle = load_semisupervised_fashion_mnist(config)
    splits = get_student_splits(bundle)

    model = ManualSoftmaxRegression(
        input_dim=splits["labeled_x"].shape[1],
        num_classes=len(bundle.class_names),
        seed=int(config["seed"]),
        init_scale=float(config["student"]["init_scale"]),
    )
    history = model.fit(
        X=splits["labeled_x"],
        hard_labels=splits["labeled_y"],
        X_val=splits["val_x"],
        y_val=splits["val_y"],
        epochs=int(config["student"]["epochs"]),
        batch_size=int(config["student"]["batch_size"]),
        learning_rate=float(config["student"]["learning_rate"]),
        weight_decay=float(config["student"]["weight_decay"]),
        seed=int(config["seed"]),
    )

    test_probs = model.predict_proba(splits["test_x"])
    metrics = evaluate_predictions(splits["test_y"], test_probs, bundle.class_names)

    artifact_dir = PROJECT_ROOT / config["output_dir"] / "artifacts" / "student_baseline"
    model.save(artifact_dir / "student_baseline.npz")
    save_json(
        {
            "train_loss": history.train_loss,
            "train_acc": history.train_acc,
            "val_acc": history.val_acc,
            "hard_loss": history.hard_loss,
            "soft_loss": history.soft_loss,
        },
        artifact_dir / "history.json",
    )
    save_json(metrics, artifact_dir / "metrics.json")
    plot_training_curves(
        {
            "train_loss": history.train_loss,
            "train_acc": history.train_acc,
            "val_acc": history.val_acc,
            "hard_loss": history.hard_loss,
            "soft_loss": history.soft_loss,
        },
        PROJECT_ROOT / config["output_dir"] / "plots" / "student_baseline_training_curves.png",
        "Student Baseline Training",
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
