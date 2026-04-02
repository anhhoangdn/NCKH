"""Simple segmentation baselines: fixed-window, shot-change, text-only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vlm_video.common.logging_utils import get_logger

logger = get_logger(__name__)


def fixed_window_segmentation(
    timestamps: list[float],
    window_sec: float = 60.0,
) -> list[dict[str, Any]]:
    """Segment by fixed time windows (ignores content).

    Parameters
    ----------
    timestamps:
        Per-frame timestamps in seconds.
    window_sec:
        Target duration of each segment in seconds.

    Returns
    -------
    list[dict]
        Segment dicts with ``start_time``, ``end_time``, ``frame_indices``.
    """
    if not timestamps:
        return []

    segments: list[dict[str, Any]] = []
    start_idx = 0
    current_start = timestamps[0]

    for i, t in enumerate(timestamps):
        if t - current_start >= window_sec:
            segments.append(
                {
                    "start_time": float(current_start),
                    "end_time": float(timestamps[i - 1]) if i > 0 else float(t),
                    "frame_indices": list(range(start_idx, i)),
                }
            )
            start_idx = i
            current_start = t

    # Final segment
    if start_idx < len(timestamps):
        segments.append(
            {
                "start_time": float(current_start),
                "end_time": float(timestamps[-1]),
                "frame_indices": list(range(start_idx, len(timestamps))),
            }
        )

    return segments


def shot_change_segmentation(
    frame_paths: list[str],
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Detect cuts using HSV histogram difference (OpenCV).

    Parameters
    ----------
    frame_paths:
        Sorted list of frame file paths.
    threshold:
        Chi-squared histogram distance threshold.  Lower values are more sensitive.

    Returns
    -------
    list[dict]
        Segment dicts with ``start_frame``, ``end_frame``, ``frame_indices``.
        Note: no time information is included because timestamps are not passed.
    """
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("opencv-python-headless is required for shot detection.") from exc

    if not frame_paths:
        return []

    def _hist(path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            return np.zeros((180, 256), dtype=np.float32)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(h, h)
        return h

    boundaries = [0]
    prev_hist = _hist(frame_paths[0])

    for i in range(1, len(frame_paths)):
        curr_hist = _hist(frame_paths[i])
        diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CHISQR)
        if diff > threshold:
            boundaries.append(i)
        prev_hist = curr_hist

    segments: list[dict[str, Any]] = []
    for j in range(len(boundaries)):
        start = boundaries[j]
        end = boundaries[j + 1] if j + 1 < len(boundaries) else len(frame_paths)
        segments.append(
            {
                "start_frame": start,
                "end_frame": end - 1,
                "frame_indices": list(range(start, end)),
            }
        )

    logger.info("Shot-change segmentation: %d segments.", len(segments))
    return segments


def text_only_segmentation(
    transcripts: list[dict[str, Any]],
    window_sec: float = 60.0,
) -> list[dict[str, Any]]:
    """Segment ASR transcripts by a sliding time window.

    Parameters
    ----------
    transcripts:
        List of ASR segment dicts with ``start`` and ``end`` keys.
    window_sec:
        Target window duration in seconds.

    Returns
    -------
    list[dict]
        Segment dicts with ``start_time``, ``end_time``, ``transcript_indices``.
    """
    if not transcripts:
        return []

    segments: list[dict[str, Any]] = []
    window_start = transcripts[0].get("start", 0.0)
    seg_indices: list[int] = []

    for i, seg in enumerate(transcripts):
        seg_indices.append(i)
        if seg.get("end", 0.0) - window_start >= window_sec:
            segments.append(
                {
                    "start_time": float(window_start),
                    "end_time": float(seg.get("end", 0.0)),
                    "transcript_indices": seg_indices[:],
                }
            )
            seg_indices = []
            if i + 1 < len(transcripts):
                window_start = transcripts[i + 1].get("start", seg.get("end", 0.0))

    if seg_indices:
        segments.append(
            {
                "start_time": float(window_start),
                "end_time": float(transcripts[-1].get("end", window_start)),
                "transcript_indices": seg_indices,
            }
        )

    return segments
