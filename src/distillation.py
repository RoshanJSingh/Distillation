from __future__ import annotations

import numpy as np

from src.semantic_message import build_semantic_message, decode_semantic_message
from src.student_manual import softmax


def teacher_probs_with_temperature(teacher_logits: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(teacher_logits, temperature=temperature)


def build_distillation_targets(
    teacher_logits: np.ndarray,
    semantic_mode: str,
    top_k: int,
    temperature: float,
) -> tuple[np.ndarray, dict[str, np.ndarray | str]]:
    teacher_probs_T = teacher_probs_with_temperature(teacher_logits, temperature=temperature)
    message = build_semantic_message(teacher_probs_T, mode=semantic_mode, top_k=top_k)
    dense_targets = decode_semantic_message(message, num_classes=teacher_probs_T.shape[1])
    return dense_targets, message


def build_pseudo_labels(teacher_probs: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    confidences = teacher_probs.max(axis=1)
    pseudo_labels = teacher_probs.argmax(axis=1)
    hard_mask = confidences >= threshold
    return pseudo_labels.astype(np.int64), hard_mask.astype(np.float32), confidences.astype(np.float32)
