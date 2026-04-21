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

from src.coreset import (
    combined_selection_score,
    js_divergence,
    predictive_entropy,
    project_embeddings,
    select_kmeans_coreset,
)
from src.data import get_student_splits, load_semisupervised_fashion_mnist, set_seed
from src.metrics import plot_projection, plot_uncertainty_histograms, save_json
from src.student_manual import ManualSoftmaxRegression
from src.teacher_model import load_teacher_checkpoint, predict_teacher


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select an unlabeled coreset with K-means++ and uncertainty scores.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--coreset-size", type=int, default=None)
    parser.add_argument("--clusters", type=int, default=None)
    parser.add_argument("--projection", type=str, default=None, choices=["pca", "tsne"])
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    if args.coreset_size is not None:
        config["coreset"]["size"] = args.coreset_size
    if args.clusters is not None:
        config["coreset"]["n_clusters"] = args.clusters
    if args.projection is not None:
        config["coreset"]["projection"] = args.projection

    set_seed(int(config["seed"]))
    bundle = load_semisupervised_fashion_mnist(config)
    splits = get_student_splits(bundle)

    teacher_path = PROJECT_ROOT / config["output_dir"] / "artifacts" / "teacher" / "best_teacher.pt"
    student_path = PROJECT_ROOT / config["output_dir"] / "artifacts" / "student_baseline" / "student_baseline.npz"
    if not teacher_path.exists():
        raise FileNotFoundError("Teacher checkpoint not found. Run scripts/train_teacher.py first.")
    if not student_path.exists():
        raise FileNotFoundError("Baseline student checkpoint not found. Run scripts/train_student_baseline.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_teacher_checkpoint(teacher_path, device)
    teacher_outputs = predict_teacher(
        teacher,
        bundle.unlabeled.images,
        batch_size=int(config["teacher"]["batch_size"]),
        device=device,
    )

    baseline_student = ManualSoftmaxRegression.load(student_path)
    student_probs = baseline_student.predict_proba(splits["unlabeled_x"])

    entropy = predictive_entropy(teacher_outputs["probs"])
    disagreement = js_divergence(teacher_outputs["probs"], student_probs)
    combined = combined_selection_score(
        entropy,
        disagreement,
        entropy_weight=float(config["coreset"]["entropy_weight"]),
        disagreement_weight=float(config["coreset"]["disagreement_weight"]),
    )

    selection = select_kmeans_coreset(
        embeddings=teacher_outputs["embeddings"],
        scores=combined,
        coreset_size=int(config["coreset"]["size"]),
        n_clusters=int(config["coreset"]["n_clusters"]),
        seed=int(config["seed"]),
    )

    selected_mask = np.zeros(len(bundle.unlabeled.images), dtype=bool)
    selected_mask[selection["selected_indices"]] = True
    projection = project_embeddings(
        teacher_outputs["embeddings"],
        method=config["coreset"]["projection"],
        seed=int(config["seed"]),
    )

    artifact_dir = PROJECT_ROOT / config["output_dir"] / "artifacts" / "coreset"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        artifact_dir / "selection.npz",
        selected_indices=selection["selected_indices"],
        cluster_ids=selection["cluster_ids"],
        original_indices=bundle.unlabeled.indices[selection["selected_indices"]],
        entropy=entropy,
        disagreement=disagreement,
        combined_score=combined,
    )
    save_json(
        {
            "selected_size": int(len(selection["selected_indices"])),
            "entropy_mean": float(entropy.mean()),
            "disagreement_mean": float(disagreement.mean()),
            "selection_score_mean": float(combined.mean()),
        },
        artifact_dir / "selection_summary.json",
    )

    plot_uncertainty_histograms(
        {"entropy": entropy, "disagreement": disagreement, "combined_score": combined},
        PROJECT_ROOT / config["output_dir"] / "plots" / "uncertainty_distributions.png",
    )
    plot_projection(
        projection,
        selected_mask,
        PROJECT_ROOT / config["output_dir"] / "plots" / "coreset_projection.png",
        f"Unlabeled Embeddings ({config['coreset']['projection'].upper()})",
    )

    print(
        json.dumps(
            {
                "selection_path": str(artifact_dir / "selection.npz"),
                "selected_size": int(len(selection["selected_indices"])),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
