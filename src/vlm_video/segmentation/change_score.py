"""Cosine change-score computation and score smoothing."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def cosine_change_score(embeddings: np.ndarray) -> np.ndarray:
    """Compute per-frame cosine change scores.

    The change score at position *t* is defined as::

        d_t = 1 - cosine_similarity(e_t, e_{t-1})

    The first score (index 0) is set to 0 because there is no previous frame.

    Parameters
    ----------
    embeddings:
        Array of shape ``(T, D)`` containing L2-normalised frame embeddings.

    Returns
    -------
    np.ndarray
        1-D array of shape ``(T,)`` with values in ``[0, 2]``.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (T × D), got shape {embeddings.shape}")

    T = embeddings.shape[0]
    scores = np.zeros(T, dtype=np.float32)
    if T < 2:
        return scores

    # Dot product of consecutive L2-normalised vectors = cosine similarity
    cos_sim = np.einsum("td,td->t", embeddings[1:], embeddings[:-1])
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    scores[1:] = 1.0 - cos_sim
    return scores


def smooth_scores(
    scores: np.ndarray,
    window: int = 3,
    method: str = "mean",
) -> np.ndarray:
    """Apply a sliding-window filter to *scores*.

    Parameters
    ----------
    scores:
        1-D array of change scores.
    window:
        Kernel width (must be odd and >= 1).  If <= 1, *scores* is returned
        unchanged.
    method:
        Aggregation method: ``"mean"`` or ``"median"``.

    Returns
    -------
    np.ndarray
        Smoothed scores array of the same length as *scores*.
    """
    if window <= 1 or len(scores) < window:
        return scores.copy()

    pad = window // 2
    padded = np.pad(scores, pad, mode="edge")
    windows = sliding_window_view(padded, window_shape=window)

    if method == "median":
        return np.median(windows, axis=1).astype(np.float32)
    return np.mean(windows, axis=1).astype(np.float32)
