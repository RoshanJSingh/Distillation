from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    safe_probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(safe_probs * np.log(safe_probs), axis=1)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    safe_p = np.clip(p, 1e-12, 1.0)
    safe_q = np.clip(q, 1e-12, 1.0)
    return np.sum(safe_p * (np.log(safe_p) - np.log(safe_q)), axis=1)


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    mean_dist = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, mean_dist) + 0.5 * kl_divergence(q, mean_dist)


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def combined_selection_score(
    entropy: np.ndarray,
    disagreement: np.ndarray,
    entropy_weight: float,
    disagreement_weight: float,
) -> np.ndarray:
    entropy_norm = minmax_normalize(entropy)
    disagreement_norm = minmax_normalize(disagreement)
    return entropy_weight * entropy_norm + disagreement_weight * disagreement_norm


def select_kmeans_coreset(
    embeddings: np.ndarray,
    scores: np.ndarray,
    coreset_size: int,
    n_clusters: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if coreset_size > len(embeddings):
        raise ValueError("coreset_size cannot exceed the unlabeled pool size.")

    cluster_count = min(n_clusters, coreset_size, len(embeddings))
    kmeans = KMeans(n_clusters=cluster_count, init="k-means++", n_init=10, random_state=seed)
    cluster_ids = kmeans.fit_predict(embeddings)

    ranked_indices = {}
    for cluster_id in range(cluster_count):
        members = np.where(cluster_ids == cluster_id)[0]
        sorted_members = members[np.argsort(scores[members])[::-1]]
        ranked_indices[cluster_id] = sorted_members.tolist()

    selected = []
    pointers = {cluster_id: 0 for cluster_id in range(cluster_count)}

    while len(selected) < coreset_size:
        progress = False
        for cluster_id in range(cluster_count):
            ranking = ranked_indices[cluster_id]
            pointer = pointers[cluster_id]
            if pointer < len(ranking):
                selected.append(ranking[pointer])
                pointers[cluster_id] += 1
                progress = True
            if len(selected) >= coreset_size:
                break
        if not progress:
            break

    selected = np.array(selected[:coreset_size], dtype=np.int64)
    return {
        "selected_indices": selected,
        "cluster_ids": cluster_ids.astype(np.int64),
        "cluster_centers": kmeans.cluster_centers_.astype(np.float32),
    }


def project_embeddings(embeddings: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "pca":
        projector = PCA(n_components=2, random_state=seed)
        return projector.fit_transform(embeddings)
    if method == "tsne":
        projector = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=seed)
        return projector.fit_transform(embeddings)
    raise ValueError(f"Unknown projection method: {method}")
