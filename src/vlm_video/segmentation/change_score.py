"""Cosine change-score computation and score smoothing."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def cosine_change_score(embeddings: np.ndarray, window: int = 5) -> np.ndarray:
    """Compute per-frame cosine change scores using windowed means.

    The change score at position *t* compares the mean embeddings in two windows::

        left  = mean(embeddings[t - w : t])
        right = mean(embeddings[t : t + w])
        d_t   = 1 - cosine_similarity(left, right)

    Scores are only computed for indices with full windows.  Other positions are
    set to 0.

    Parameters
    ----------
    embeddings:
        Array of shape ``(T, D)`` containing L2-normalised frame embeddings.
    window:
        Window size *w* for the mean comparison.

    Returns
    -------
    np.ndarray
        1-D array of shape ``(T,)`` with values in ``[0, 2]``.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (T × D), got shape {embeddings.shape}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    T, dim = embeddings.shape
    scores = np.zeros(T, dtype=np.float32)
    if T < 2 * window:
        return scores

    cumsum = np.zeros((T + 1, dim), dtype=np.float64)
    cumsum[1:] = np.cumsum(embeddings, axis=0)

    idx = np.arange(window, T - window + 1)
    left = (cumsum[idx] - cumsum[idx - window]) / float(window)
    right = (cumsum[idx + window] - cumsum[idx]) / float(window)

    dot = np.einsum("nd,nd->n", left, right)
    denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cos_sim = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 1e-10)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    scores[idx] = 1.0 - cos_sim
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
