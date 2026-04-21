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
from src.distillation import build_distillation_targets, build_pseudo_labels
from src.metrics import evaluate_predictions, plot_training_curves, save_json
from src.student_manual import ManualSoftmaxRegression
from src.teacher_model import load_teacher_checkpoint, predict_teacher


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def history_to_dict(history) -> dict[str, list[float]]:
    return {
        "train_loss": history.train_loss,
        "train_acc": history.train_acc,
        "val_acc": history.val_acc,
        "hard_loss": history.hard_loss,
        "soft_loss": history.soft_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a manual student with distillation.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--mode", type=str, choices=["distill_only", "coreset"], default="distill_only")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--semantic-mode", type=str, choices=["full", "topk"], default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--pseudo-threshold", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    if args.alpha is not None:
        config["distillation"]["alpha"] = args.alpha
    if args.temperature is not None:
        config["distillation"]["temperature"] = args.temperature
    if args.semantic_mode is not None:
        config["distillation"]["semantic_mode"] = args.semantic_mode
    if args.top_k is not None:
        config["distillation"]["top_k"] = args.top_k
    if args.pseudo_threshold is not None:
        config["distillation"]["pseudo_label_threshold"] = args.pseudo_threshold
    if args.epochs is not None:
        config["student"]["epochs"] = args.epochs
    if args.learning_rate is not None:
        config["student"]["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        config["student"]["batch_size"] = args.batch_size

    set_seed(int(config["seed"]))
    bundle = load_semisupervised_fashion_mnist(config)
    splits = get_student_splits(bundle)

    teacher_path = PROJECT_ROOT / config["output_dir"] / "artifacts" / "teacher" / "best_teacher.pt"
    if not teacher_path.exists():
        raise FileNotFoundError("Teacher checkpoint not found. Run scripts/train_teacher.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_teacher_checkpoint(teacher_path, device)

    labeled_teacher_outputs = predict_teacher(
        teacher,
        bundle.labeled.images,
        batch_size=int(config["teacher"]["batch_size"]),
        device=device,
    )
    labeled_soft_targets, labeled_message = build_distillation_targets(
        teacher_logits=labeled_teacher_outputs["logits"],
        semantic_mode=config["distillation"]["semantic_mode"],
        top_k=int(config["distillation"]["top_k"]),
        temperature=float(config["distillation"]["temperature"]),
    )

    train_x = splits["labeled_x"]
    train_hard = splits["labeled_y"]
    train_soft = labeled_soft_targets
    hard_mask = np.ones(train_hard.shape[0], dtype=np.float32)
    pseudo_summary = {"pseudo_label_count": 0, "semantic_message_values": int(np.prod(labeled_message["probabilities"].shape))}

    experiment_name = "student_distill_only"
    plot_name = "student_distill_only_training_curves.png"

    if args.mode == "coreset":
        coreset_path = PROJECT_ROOT / config["output_dir"] / "artifacts" / "coreset" / "selection.npz"
        if not coreset_path.exists():
            raise FileNotFoundError("Coreset file not found. Run scripts/select_coreset.py first.")

        selection_payload = np.load(coreset_path)
        selected_indices = selection_payload["selected_indices"].astype(np.int64)

        coreset_images = bundle.unlabeled.images[selected_indices]
        coreset_x = splits["unlabeled_x"][selected_indices]
        coreset_teacher_outputs = predict_teacher(
            teacher,
            coreset_images,
            batch_size=int(config["teacher"]["batch_size"]),
            device=device,
        )
        coreset_soft_targets, coreset_message = build_distillation_targets(
            teacher_logits=coreset_teacher_outputs["logits"],
            semantic_mode=config["distillation"]["semantic_mode"],
            top_k=int(config["distillation"]["top_k"]),
            temperature=float(config["distillation"]["temperature"]),
        )
        pseudo_labels, pseudo_mask, confidences = build_pseudo_labels(
            coreset_teacher_outputs["probs"],
            threshold=float(config["distillation"]["pseudo_label_threshold"]),
        )

        train_x = np.concatenate([train_x, coreset_x], axis=0)
        train_hard = np.concatenate([train_hard, pseudo_labels], axis=0)
        train_soft = np.concatenate([train_soft, coreset_soft_targets], axis=0)
        hard_mask = np.concatenate([hard_mask, pseudo_mask], axis=0)

        pseudo_summary = {
            "pseudo_label_count": int(pseudo_mask.sum()),
            "pseudo_label_fraction": float(pseudo_mask.mean()),
            "mean_teacher_confidence": float(confidences.mean()),
            "semantic_message_values": int(
                np.prod(labeled_message["probabilities"].shape) + np.prod(coreset_message["probabilities"].shape)
            ),
        }
        experiment_name = "student_distill_coreset"
        plot_name = "student_distill_coreset_training_curves.png"

    model = ManualSoftmaxRegression(
        input_dim=train_x.shape[1],
        num_classes=len(bundle.class_names),
        seed=int(config["seed"]),
        init_scale=float(config["student"]["init_scale"]),
    )
    history = model.fit(
        X=train_x,
        hard_labels=train_hard,
        X_val=splits["val_x"],
        y_val=splits["val_y"],
        epochs=int(config["student"]["epochs"]),
        batch_size=int(config["student"]["batch_size"]),
        learning_rate=float(config["student"]["learning_rate"]),
        weight_decay=float(config["student"]["weight_decay"]),
        soft_targets=train_soft,
        alpha=float(config["distillation"]["alpha"]),
        temperature=float(config["distillation"]["temperature"]),
        hard_mask=hard_mask,
        seed=int(config["seed"]),
    )

    test_probs = model.predict_proba(splits["test_x"])
    metrics = evaluate_predictions(splits["test_y"], test_probs, bundle.class_names)
    metrics["pseudo_summary"] = pseudo_summary
    metrics["mode"] = args.mode

    artifact_dir = PROJECT_ROOT / config["output_dir"] / "artifacts" / experiment_name
    model.save(artifact_dir / f"{experiment_name}.npz")
    save_json(history_to_dict(history), artifact_dir / "history.json")
    save_json(metrics, artifact_dir / "metrics.json")
    plot_training_curves(
        history_to_dict(history),
        PROJECT_ROOT / config["output_dir"] / "plots" / plot_name,
        experiment_name.replace("_", " ").title(),
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
