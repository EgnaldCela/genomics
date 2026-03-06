from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from sklearn.manifold import TSNE

TensorMetric = Callable[[torch.Tensor, torch.Tensor], float | torch.Tensor]
ArrayLike = torch.Tensor | np.ndarray


def _to_float(value: float | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def sanitize_distance_matrix(dist_matrix: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf values with finite fallbacks for downstream manifold learning."""
    return np.nan_to_num(dist_matrix, nan=0.0, posinf=1e6, neginf=0.0)


def compute_pairwise_distance_matrix(X: ArrayLike, metric_func: TensorMetric) -> np.ndarray:
    """Compute a symmetric pairwise distance matrix from histogram-like vectors."""
    X_tensor = torch.as_tensor(X)
    n = X_tensor.shape[0]
    dist_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            d = _to_float(metric_func(X_tensor[i], X_tensor[j]))
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return sanitize_distance_matrix(dist_matrix)


def compute_tsne_embedding(
    X: ArrayLike,
    *,
    metric: str,
    random_state: int = 8,
) -> np.ndarray:
    """
    Compute a 2D t-SNE embedding.

    `metric` may be `euclidean` (directly on vectors) or `precomputed` (on a square matrix).
    """
    tsne = TSNE(n_components=2, metric=metric, init="random", random_state=random_state)
    if metric == "euclidean":
        X_array = torch.as_tensor(X).numpy()
        return tsne.fit_transform(X_array)

    X_array = np.asarray(X)
    return tsne.fit_transform(X_array)
