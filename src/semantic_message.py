from __future__ import annotations

import numpy as np


def build_semantic_message(probabilities: np.ndarray, mode: str = "full", top_k: int = 3) -> dict[str, np.ndarray | str]:
    probs = np.atleast_2d(probabilities).astype(np.float32)

    if mode == "full":
        return {"mode": "full", "probabilities": probs}

    if mode != "topk":
        raise ValueError(f"Unsupported semantic mode: {mode}")

    top_k = min(top_k, probs.shape[1])
    top_indices = np.argpartition(-probs, kth=top_k - 1, axis=1)[:, :top_k]
    top_probs = np.take_along_axis(probs, top_indices, axis=1)
    order = np.argsort(-top_probs, axis=1)
    top_indices = np.take_along_axis(top_indices, order, axis=1)
    top_probs = np.take_along_axis(top_probs, order, axis=1)
    renormalized = top_probs / np.clip(top_probs.sum(axis=1, keepdims=True), 1e-12, None)
    return {
        "mode": "topk",
        "class_ids": top_indices.astype(np.int64),
        "probabilities": renormalized.astype(np.float32),
    }


def decode_semantic_message(message: dict[str, np.ndarray | str], num_classes: int) -> np.ndarray:
    mode = message["mode"]
    if mode == "full":
        return np.asarray(message["probabilities"], dtype=np.float32)

    if mode == "topk":
        decoded = np.zeros((message["class_ids"].shape[0], num_classes), dtype=np.float32)
        rows = np.arange(decoded.shape[0])[:, None]
        decoded[rows, message["class_ids"]] = message["probabilities"]
        return decoded

    raise ValueError(f"Unsupported semantic mode: {mode}")


def semantic_message_size(message: dict[str, np.ndarray | str]) -> int:
    mode = message["mode"]
    if mode == "full":
        return int(np.asarray(message["probabilities"]).size)
    return int(np.asarray(message["class_ids"]).size + np.asarray(message["probabilities"]).size)
