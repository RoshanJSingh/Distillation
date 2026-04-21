from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST

FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


@dataclass
class SplitData:
    images: np.ndarray
    labels: Optional[np.ndarray]
    indices: np.ndarray
    hidden_labels: Optional[np.ndarray] = None


@dataclass
class DataBundle:
    labeled: SplitData
    unlabeled: SplitData
    val: SplitData
    test: SplitData
    class_names: list[str]
    student_feature_mean: np.ndarray
    student_feature_std: np.ndarray


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_fashion_mnist(root: str | Path, train: bool, download: bool = True) -> tuple[np.ndarray, np.ndarray]:
    dataset = FashionMNIST(root=str(root), train=train, download=download)
    images = dataset.data.numpy().astype(np.float32) / 255.0
    images = np.expand_dims(images, axis=1)
    labels = dataset.targets.numpy().astype(np.int64)
    return images, labels


def load_semisupervised_fashion_mnist(config: dict) -> DataBundle:
    seed = int(config["seed"])
    data_cfg = config["data"]
    rng = np.random.default_rng(seed)

    root = Path(data_cfg["data_root"])
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    train_images, train_labels = _load_fashion_mnist(root=root, train=True, download=True)
    test_images, test_labels = _load_fashion_mnist(root=root, train=False, download=True)

    max_train_samples = int(data_cfg["max_train_samples"])
    labeled_count = int(data_cfg["labeled_count"])
    unlabeled_count = int(data_cfg["unlabeled_count"])
    val_count = int(data_cfg["val_count"])

    if labeled_count + unlabeled_count + val_count > max_train_samples:
        raise ValueError("The labeled, unlabeled, and validation counts exceed max_train_samples.")

    shuffled_train = rng.permutation(len(train_images))
    selected = shuffled_train[:max_train_samples]

    labeled_idx = selected[:labeled_count]
    unlabeled_idx = selected[labeled_count : labeled_count + unlabeled_count]
    val_idx = selected[labeled_count + unlabeled_count : labeled_count + unlabeled_count + val_count]

    test_count = data_cfg.get("test_count")
    if test_count is not None:
        test_idx = rng.permutation(len(test_images))[: int(test_count)]
        test_images = test_images[test_idx]
        test_labels = test_labels[test_idx]
        test_indices = test_idx
    else:
        test_indices = np.arange(len(test_images), dtype=np.int64)

    labeled_images = train_images[labeled_idx]
    labeled_labels = train_labels[labeled_idx]

    feature_mean, feature_std = compute_student_normalization(labeled_images)

    return DataBundle(
        labeled=SplitData(images=labeled_images, labels=labeled_labels, indices=labeled_idx),
        unlabeled=SplitData(
            images=train_images[unlabeled_idx],
            labels=None,
            indices=unlabeled_idx,
            hidden_labels=train_labels[unlabeled_idx],
        ),
        val=SplitData(images=train_images[val_idx], labels=train_labels[val_idx], indices=val_idx),
        test=SplitData(images=test_images, labels=test_labels, indices=test_indices),
        class_names=FASHION_MNIST_CLASSES,
        student_feature_mean=feature_mean,
        student_feature_std=feature_std,
    )


def compute_student_normalization(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flattened = images.reshape(images.shape[0], -1)
    mean = flattened.mean(axis=0)
    std = flattened.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def flatten_for_student(images: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    flattened = images.reshape(images.shape[0], -1).astype(np.float32)
    return (flattened - mean) / std


def make_loader(
    images: np.ndarray,
    labels: Optional[np.ndarray],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    image_tensor = torch.from_numpy(images).float()
    if labels is None:
        dataset = TensorDataset(image_tensor)
    else:
        label_tensor = torch.from_numpy(labels).long()
        dataset = TensorDataset(image_tensor, label_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_teacher_loaders(bundle: DataBundle, config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    teacher_cfg = config["teacher"]
    num_workers = int(config["data"].get("num_workers", 0))
    batch_size = int(teacher_cfg["batch_size"])
    train_loader = make_loader(bundle.labeled.images, bundle.labeled.labels, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_loader(bundle.val.images, bundle.val.labels, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = make_loader(bundle.test.images, bundle.test.labels, batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def get_student_splits(bundle: DataBundle) -> dict[str, np.ndarray]:
    return {
        "labeled_x": flatten_for_student(bundle.labeled.images, bundle.student_feature_mean, bundle.student_feature_std),
        "labeled_y": bundle.labeled.labels.copy(),
        "unlabeled_x": flatten_for_student(bundle.unlabeled.images, bundle.student_feature_mean, bundle.student_feature_std),
        "val_x": flatten_for_student(bundle.val.images, bundle.student_feature_mean, bundle.student_feature_std),
        "val_y": bundle.val.labels.copy(),
        "test_x": flatten_for_student(bundle.test.images, bundle.student_feature_mean, bundle.student_feature_std),
        "test_y": bundle.test.labels.copy(),
    }
