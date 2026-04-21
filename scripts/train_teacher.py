from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import get_teacher_loaders, load_semisupervised_fashion_mnist, set_seed
from src.metrics import plot_training_curves, save_json
from src.teacher_model import SmallCNNTeacher, train_teacher


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Fashion-MNIST teacher CNN.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    if args.epochs is not None:
        config["teacher"]["epochs"] = args.epochs
    if args.learning_rate is not None:
        config["teacher"]["learning_rate"] = args.learning_rate

    set_seed(int(config["seed"]))
    bundle = load_semisupervised_fashion_mnist(config)
    train_loader, val_loader, _ = get_teacher_loaders(bundle, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNNTeacher(embedding_dim=int(config["teacher"]["embedding_dim"])).to(device)

    artifact_dir = PROJECT_ROOT / config["output_dir"] / "artifacts" / "teacher"
    checkpoint_path = artifact_dir / "best_teacher.pt"
    history = train_teacher(model, train_loader, val_loader, config, device, checkpoint_path)

    plot_training_curves(history, PROJECT_ROOT / config["output_dir"] / "plots" / "teacher_training_curves.png", "Teacher Training")
    save_json(history, artifact_dir / "teacher_history.json")

    print(json.dumps({"checkpoint": str(checkpoint_path), "best_val_acc": max(history["val_acc"])}, indent=2))


if __name__ == "__main__":
    main()
