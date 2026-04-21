from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import get_student_splits, load_semisupervised_fashion_mnist, set_seed
from src.metrics import (
    evaluate_predictions,
    plot_accuracy_bar,
    plot_confusion_matrix,
    save_json,
    save_results_csv,
)
from src.student_manual import ManualSoftmaxRegression
from src.teacher_model import load_teacher_checkpoint, predict_teacher


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate teacher and all student variants.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    set_seed(int(config["seed"]))
    bundle = load_semisupervised_fashion_mnist(config)
    splits = get_student_splits(bundle)

    teacher_path = PROJECT_ROOT / config["output_dir"] / "artifacts" / "teacher" / "best_teacher.pt"
    baseline_path = PROJECT_ROOT / config["output_dir"] / "artifacts" / "student_baseline" / "student_baseline.npz"
    distill_only_path = PROJECT_ROOT / config["output_dir"] / "artifacts" / "student_distill_only" / "student_distill_only.npz"
    distill_coreset_path = PROJECT_ROOT / config["output_dir"] / "artifacts" / "student_distill_coreset" / "student_distill_coreset.npz"

    required_paths = [teacher_path, baseline_path, distill_only_path, distill_coreset_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_teacher_checkpoint(teacher_path, device)
    teacher_outputs = predict_teacher(
        teacher,
        bundle.test.images,
        batch_size=int(config["teacher"]["batch_size"]),
        device=device,
    )

    baseline = ManualSoftmaxRegression.load(baseline_path)
    distill_only = ManualSoftmaxRegression.load(distill_only_path)
    distill_coreset = ManualSoftmaxRegression.load(distill_coreset_path)

    results = {
        "teacher": evaluate_predictions(bundle.test.labels, teacher_outputs["probs"], bundle.class_names),
        "student_baseline": evaluate_predictions(bundle.test.labels, baseline.predict_proba(splits["test_x"]), bundle.class_names),
        "student_distill_only": evaluate_predictions(bundle.test.labels, distill_only.predict_proba(splits["test_x"]), bundle.class_names),
        "student_distill_coreset": evaluate_predictions(bundle.test.labels, distill_coreset.predict_proba(splits["test_x"]), bundle.class_names),
    }

    results_dir = PROJECT_ROOT / config["output_dir"] / "results"
    save_json(results, results_dir / "summary.json")
    save_results_csv(results, results_dir / "summary.csv")
    plot_accuracy_bar(results, PROJECT_ROOT / config["output_dir"] / "plots" / "experiment_comparison.png")

    for setting, metrics in results.items():
        cm = np.array(metrics["confusion_matrix"], dtype=np.int64)
        plot_confusion_matrix(
            cm,
            bundle.class_names,
            PROJECT_ROOT / config["output_dir"] / "plots" / f"{setting}_confusion_matrix.png",
            title=f"{setting} confusion matrix",
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
