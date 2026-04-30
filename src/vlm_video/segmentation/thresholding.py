"""Threshold-based boundary detection and segment post-processing."""

from __future__ import annotations

from typing import Any

import numpy as np


def fixed_threshold(scores: np.ndarray, threshold: float) -> list[int]:
    """Return frame indices where the change score exceeds *threshold*.

    Parameters
    ----------
    scores:
        1-D array of change scores (output of
        :func:`~vlm_video.segmentation.change_score.cosine_change_score`).
    threshold:
        Minimum score to qualify as a boundary.

    Returns
    -------
    list[int]
        Sorted list of frame indices that are segment boundaries.
    """
    return [int(i) for i in np.where(scores > threshold)[0]]


def otsu_threshold(scores: np.ndarray) -> list[int]:
    """Return frame indices where the score exceeds an Otsu-derived threshold.

    Falls back to mean + 0.5 * std if Otsu computation fails.
    """
    if len(scores) == 0:
        return []
    try:
        from skimage.filters import threshold_otsu

        thresh = float(threshold_otsu(scores))
        if np.isnan(thresh):
            raise ValueError("Otsu threshold returned NaN.")
    except Exception:
        thresh = float(np.mean(scores) + 0.5 * np.std(scores))

    return fixed_threshold(scores, threshold=thresh)


def adaptive_threshold(scores: np.ndarray, percentile: float = 85) -> list[int]:
    """Return frame indices where the change score exceeds the *percentile* value.

    Parameters
    ----------
    scores:
        1-D array of change scores.
    percentile:
        Percentile of the score distribution used as the dynamic threshold
        (e.g. 85 means the top 15 % of scores are boundaries).

    Returns
    -------
    list[int]
        Sorted list of boundary frame indices.
    """
    if len(scores) == 0:
        return []
    thresh = float(np.percentile(scores, percentile))
    return fixed_threshold(scores, threshold=thresh)


def enforce_min_duration(
    boundaries: list[int],
    timestamps: list[float],
    min_sec: float = 5.0,
) -> list[int]:
    """Remove boundary indices that are too close to the previous one.

    Parameters
    ----------
    boundaries:
        Sorted list of frame indices marking segment boundaries.
    timestamps:
        Per-frame timestamps in seconds (same length as the embeddings array).
    min_sec:
        Minimum allowed gap (seconds) between consecutive boundaries.

    Returns
    -------
    list[int]
        Filtered list of boundary indices.
    """
    if not boundaries or len(timestamps) == 0:
        return boundaries

    filtered: list[int] = []
    last_time = 0.0

    for idx in sorted(boundaries):
        if idx >= len(timestamps):
            continue
        t = timestamps[idx]
        if t - last_time >= min_sec:
            filtered.append(idx)
            last_time = t

    return filtered


def merge_short_segments(
    segments: list[dict[str, Any]],
    min_duration_sec: float,
) -> list[dict[str, Any]]:
    """Merge segments shorter than *min_duration_sec* with an adjacent neighbor."""
    if len(segments) <= 1:
        return segments

    merged: list[dict[str, Any]] = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        duration = float(seg.get("end_time", 0.0) - seg.get("start_time", 0.0))

        if duration >= min_duration_sec:
            merged.append(seg)
            i += 1
            continue

        if merged:
            prev = merged[-1]
            prev["end_time"] = seg["end_time"]
            prev["frame_indices"] = prev["frame_indices"] + seg["frame_indices"]
            i += 1
            continue

        if i + 1 < len(segments):
            nxt = segments[i + 1]
            nxt["start_time"] = seg["start_time"]
            nxt["frame_indices"] = seg["frame_indices"] + nxt["frame_indices"]
            i += 1
            continue

        merged.append(seg)
        i += 1

    return merged


def merge_segments(
    segments: list[dict[str, Any]],
    embeddings: np.ndarray,
    sim_threshold: float = 0.9,
) -> list[dict[str, Any]]:
    """Merge adjacent segments whose mean embeddings are very similar.

    Parameters
    ----------
    segments:
        List of segment dicts with at least ``frame_indices``.
    embeddings:
        Array of shape ``(T, D)`` containing per-frame embeddings.
    sim_threshold:
        Cosine similarity threshold above which two adjacent segments are merged.

    Returns
    -------
    list[dict]
        Merged segment list (same format as input).
    """
    if len(segments) <= 1:
        return segments

    def mean_emb(seg: dict[str, Any]) -> np.ndarray:
        indices = seg.get("frame_indices", [])
        if not indices:
            return np.zeros(embeddings.shape[1], dtype=np.float32)
        vecs = embeddings[indices]
        m = vecs.mean(axis=0)
        norm = np.linalg.norm(m)
        return m / norm if norm > 1e-10 else m

    merged: list[dict[str, Any]] = [segments[0]]

    for seg in segments[1:]:
        prev = merged[-1]
        cos = float(np.dot(mean_emb(prev), mean_emb(seg)))
        if cos >= sim_threshold:
            prev["end_time"] = seg["end_time"]
            prev["frame_indices"] = prev["frame_indices"] + seg["frame_indices"]
        else:
            merged.append(seg)

    return merged
