from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNNTeacher(nn.Module):
    def __init__(self, embedding_dim: int = 64, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding_layer = nn.Linear(64, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.features(x).flatten(1)
        embedding = F.relu(self.embedding_layer(features))
        logits = self.classifier(embedding)
        probs = torch.softmax(logits, dim=1)
        return {"logits": logits, "probs": probs, "embedding": embedding}


def _run_epoch(
    model: SmallCNNTeacher,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        logits = outputs["logits"]
        loss = criterion(logits, labels)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def train_teacher(
    model: SmallCNNTeacher,
    train_loader,
    val_loader,
    config: dict,
    device: torch.device,
    checkpoint_path: str | Path,
) -> dict[str, list[float]]:
    teacher_cfg = config["teacher"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(teacher_cfg["learning_rate"]),
        weight_decay=float(teacher_cfg["weight_decay"]),
    )
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_acc = -np.inf

    for _ in range(int(teacher_cfg["epochs"])):
        train_loss, train_acc = _run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, device, optimizer=None)

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "embedding_dim": int(teacher_cfg["embedding_dim"]),
                    "num_classes": 10,
                    "best_val_acc": float(best_val_acc),
                },
                checkpoint_path,
            )

    return history


@torch.no_grad()
def predict_teacher(model: SmallCNNTeacher, images: np.ndarray, batch_size: int, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    tensor = torch.from_numpy(images).float()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensor), batch_size=batch_size, shuffle=False)

    logits_list = []
    probs_list = []
    embedding_list = []
    for (batch_images,) in loader:
        batch_images = batch_images.to(device)
        outputs = model(batch_images)
        logits_list.append(outputs["logits"].cpu().numpy())
        probs_list.append(outputs["probs"].cpu().numpy())
        embedding_list.append(outputs["embedding"].cpu().numpy())

    return {
        "logits": np.concatenate(logits_list, axis=0),
        "probs": np.concatenate(probs_list, axis=0),
        "embeddings": np.concatenate(embedding_list, axis=0),
    }


def load_teacher_checkpoint(checkpoint_path: str | Path, device: torch.device) -> SmallCNNTeacher:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SmallCNNTeacher(
        embedding_dim=int(checkpoint["embedding_dim"]),
        num_classes=int(checkpoint["num_classes"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
