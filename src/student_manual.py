from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_scores = np.exp(scaled)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


@dataclass
class TrainingHistory:
    train_loss: list[float]
    train_acc: list[float]
    val_acc: list[float]
    hard_loss: list[float]
    soft_loss: list[float]


class ManualSoftmaxRegression:
    def __init__(self, input_dim: int, num_classes: int, seed: int = 42, init_scale: float = 0.01) -> None:
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.W = rng.normal(loc=0.0, scale=init_scale, size=(input_dim, num_classes)).astype(np.float32)
        self.b = np.zeros(num_classes, dtype=np.float32)

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W + self.b

    def predict_proba(self, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        return softmax(self.predict_logits(X), temperature=temperature)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W=self.W, b=self.b, input_dim=self.input_dim, num_classes=self.num_classes)

    @classmethod
    def load(cls, path: str | Path) -> "ManualSoftmaxRegression":
        payload = np.load(path)
        model = cls(
            input_dim=int(payload["input_dim"]),
            num_classes=int(payload["num_classes"]),
            seed=0,
            init_scale=0.0,
        )
        model.W = payload["W"].astype(np.float32)
        model.b = payload["b"].astype(np.float32)
        return model

    def _loss_and_gradients(
        self,
        X: np.ndarray,
        hard_labels: Optional[np.ndarray] = None,
        soft_targets: Optional[np.ndarray] = None,
        alpha: float = 1.0,
        temperature: float = 1.0,
        hard_mask: Optional[np.ndarray] = None,
        weight_decay: float = 0.0,
    ) -> dict[str, np.ndarray | float]:
        logits = self.predict_logits(X)
        probs = softmax(logits)

        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)
        hard_loss = 0.0
        soft_loss = 0.0

        if hard_labels is not None:
            hard_targets = one_hot(hard_labels.astype(np.int64), self.num_classes)
            if hard_mask is None:
                hard_mask = np.ones(hard_labels.shape[0], dtype=np.float32)
            else:
                hard_mask = hard_mask.astype(np.float32)

            active = np.maximum(hard_mask.sum(), 1.0)
            masked_probs = np.clip(probs, 1e-12, 1.0)
            hard_loss = float(
                -np.sum(hard_mask[:, None] * hard_targets * np.log(masked_probs)) / active
            )

            # For softmax regression the logit gradient is P - Q.
            grad_logits_hard = hard_mask[:, None] * (probs - hard_targets) / active
            grad_W += alpha * (X.T @ grad_logits_hard)
            grad_b += alpha * grad_logits_hard.sum(axis=0)

        if soft_targets is not None:
            student_probs_T = softmax(logits, temperature=temperature)
            safe_teacher = np.clip(soft_targets, 1e-12, 1.0)
            safe_student = np.clip(student_probs_T, 1e-12, 1.0)
            soft_loss = float(
                (temperature**2)
                * np.mean(np.sum(safe_teacher * (np.log(safe_teacher) - np.log(safe_student)), axis=1))
            )

            grad_logits_soft = temperature * (student_probs_T - soft_targets) / X.shape[0]
            grad_W += (1.0 - alpha) * (X.T @ grad_logits_soft)
            grad_b += (1.0 - alpha) * grad_logits_soft.sum(axis=0)

        if weight_decay > 0.0:
            grad_W += weight_decay * self.W

        total_loss = alpha * hard_loss + (1.0 - alpha) * soft_loss
        if hard_labels is None:
            total_loss = soft_loss
        elif soft_targets is None:
            total_loss = hard_loss

        return {
            "loss": float(total_loss),
            "hard_loss": float(hard_loss),
            "soft_loss": float(soft_loss),
            "grad_W": grad_W.astype(np.float32),
            "grad_b": grad_b.astype(np.float32),
        }

    def fit(
        self,
        X: np.ndarray,
        hard_labels: Optional[np.ndarray],
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float = 0.0,
        soft_targets: Optional[np.ndarray] = None,
        alpha: float = 1.0,
        temperature: float = 1.0,
        hard_mask: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> TrainingHistory:
        rng = np.random.default_rng(seed)
        history = TrainingHistory(train_loss=[], train_acc=[], val_acc=[], hard_loss=[], soft_loss=[])
        num_examples = X.shape[0]

        for _ in range(epochs):
            order = rng.permutation(num_examples)
            batch_losses = []
            batch_hard_losses = []
            batch_soft_losses = []

            for start in range(0, num_examples, batch_size):
                batch_idx = order[start : start + batch_size]
                batch_X = X[batch_idx]
                batch_y = None if hard_labels is None else hard_labels[batch_idx]
                batch_soft = None if soft_targets is None else soft_targets[batch_idx]
                batch_hard_mask = None if hard_mask is None else hard_mask[batch_idx]

                gradients = self._loss_and_gradients(
                    batch_X,
                    hard_labels=batch_y,
                    soft_targets=batch_soft,
                    alpha=alpha,
                    temperature=temperature,
                    hard_mask=batch_hard_mask,
                    weight_decay=weight_decay,
                )
                self.W -= learning_rate * gradients["grad_W"]
                self.b -= learning_rate * gradients["grad_b"]

                batch_losses.append(float(gradients["loss"]))
                batch_hard_losses.append(float(gradients["hard_loss"]))
                batch_soft_losses.append(float(gradients["soft_loss"]))

            train_probs = self.predict_proba(X)
            if hard_labels is not None:
                train_acc = float((train_probs.argmax(axis=1) == hard_labels).mean())
            else:
                train_acc = float("nan")
            val_acc = float((self.predict_proba(X_val).argmax(axis=1) == y_val).mean())

            history.train_loss.append(float(np.mean(batch_losses)))
            history.train_acc.append(train_acc)
            history.val_acc.append(val_acc)
            history.hard_loss.append(float(np.mean(batch_hard_losses)))
            history.soft_loss.append(float(np.mean(batch_soft_losses)))

        return history
