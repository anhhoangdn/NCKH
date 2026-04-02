"""High-level VideoSegmenter orchestrating the full segmentation pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from vlm_video.common.logging_utils import get_logger
from vlm_video.embeddings.clip_encoder import CLIPEncoder
from vlm_video.embeddings.fusion import late_fusion
from vlm_video.segmentation.change_score import cosine_change_score, smooth_scores
from vlm_video.segmentation.thresholding import (
    adaptive_threshold,
    enforce_min_duration,
    fixed_threshold,
    merge_segments,
)

logger = get_logger(__name__)


class VideoSegmenter:
    """Segment a video into topically coherent parts.

    Parameters
    ----------
    config:
        Merged configuration dictionary (from :func:`~vlm_video.common.config.load_config`).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.cfg = config
        seg_cfg = config.get("segmentation", {})
        self.method: str = seg_cfg.get("method", "clip_latefusion")
        self.threshold: float = seg_cfg.get("threshold", 0.4)
        self.adaptive_percentile: float = seg_cfg.get("adaptive_percentile", 85)
        self.min_duration: float = seg_cfg.get("min_duration", 5)
        self.smooth_window: int = seg_cfg.get("smooth_window", 3)
        self.merge_sim_threshold: float = seg_cfg.get("merge_sim_threshold", 0.9)

        emb_cfg = config.get("embeddings", {})
        self.encoder = CLIPEncoder(
            model_name=emb_cfg.get("model", "ViT-B-32"),
            pretrained=emb_cfg.get("pretrained", "laion2b_s34b_b79k"),
            device=emb_cfg.get("device", "cpu"),
        )
        w = emb_cfg.get("weights", {})
        self.wv: float = w.get("visual", 0.6)
        self.wt: float = w.get("text", 0.3)
        self.wo: float = w.get("ocr", 0.1)

    def _build_embeddings(
        self,
        frames: list[str],
        transcripts: list[dict[str, Any]] | None,
        ocr_texts: list[str] | None,
        timestamps: list[float],
    ) -> np.ndarray:
        """Build per-frame fused embeddings."""
        n = len(frames)
        emb_list: list[np.ndarray] = []

        for i, frame_path in enumerate(frames):
            t = timestamps[i] if i < len(timestamps) else 0.0

            # Visual
            vis_emb = self.encoder.encode_image(frame_path)

            # Text from ASR — find segment that overlaps frame timestamp
            text_emb = None
            if transcripts and self.wt > 0:
                text = self._text_at(t, transcripts)
                if text:
                    text_emb = self.encoder.encode_text(text)

            # OCR
            ocr_emb = None
            if ocr_texts and i < len(ocr_texts) and self.wo > 0:
                ot = ocr_texts[i]
                if ot:
                    ocr_emb = self.encoder.encode_text(ot)

            fused = late_fusion(vis_emb, text_emb, ocr_emb, self.wv, self.wt, self.wo)
            emb_list.append(fused)

        return np.stack(emb_list, axis=0)

    @staticmethod
    def _text_at(t: float, transcripts: list[dict[str, Any]]) -> str:
        """Return ASR text whose time window covers timestamp *t*."""
        for seg in transcripts:
            if seg.get("start", 0) <= t <= seg.get("end", 0):
                return seg.get("text", "")
        return ""

    def _detect_boundaries(
        self,
        embeddings: np.ndarray,
        timestamps: list[float],
    ) -> list[int]:
        scores = cosine_change_score(embeddings)
        scores = smooth_scores(scores, window=self.smooth_window)

        if self.method == "clip_latefusion":
            boundaries = adaptive_threshold(scores, self.adaptive_percentile)
        else:
            boundaries = fixed_threshold(scores, self.threshold)

        boundaries = enforce_min_duration(boundaries, timestamps, self.min_duration)
        return boundaries

    def segment(
        self,
        frames: list[str],
        timestamps: list[float],
        transcripts: list[dict[str, Any]] | None = None,
        ocr_texts: list[str] | None = None,
        embeddings: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Segment video frames into topical segments.

        Parameters
        ----------
        frames:
            List of frame file paths.
        timestamps:
            Per-frame timestamps in seconds (same length as *frames*).
        transcripts:
            Optional list of ASR segment dicts with ``start``, ``end``, ``text``.
        ocr_texts:
            Optional per-frame OCR text strings.
        embeddings:
            Pre-computed fused embeddings ``(T, D)``.  If *None*, they are
            computed on-the-fly from *frames*.

        Returns
        -------
        list[dict]
            Each segment dict contains:
            ``start_time``, ``end_time``, ``frame_indices``, ``embedding``
            (mean-pooled, as a Python list for JSON serializability).
        """
        if not frames:
            return []

        if embeddings is None:
            logger.info("Computing embeddings for %d frames …", len(frames))
            embeddings = self._build_embeddings(frames, transcripts, ocr_texts, timestamps)

        boundaries = self._detect_boundaries(embeddings, timestamps)
        logger.info("Detected %d boundaries.", len(boundaries))

        # Build segment list from boundary indices
        boundary_set = set(boundaries)
        raw_segments: list[dict[str, Any]] = []

        split_points = sorted(boundary_set | {len(frames)})
        prev = 0
        for bp in split_points:
            if bp == 0:
                continue
            indices = list(range(prev, bp))
            if not indices:
                continue
            seg_emb = embeddings[indices].mean(axis=0)
            norm = np.linalg.norm(seg_emb)
            if norm > 1e-10:
                seg_emb = seg_emb / norm

            raw_segments.append(
                {
                    "start_time": float(timestamps[prev]) if prev < len(timestamps) else 0.0,
                    "end_time": float(timestamps[bp - 1]) if (bp - 1) < len(timestamps) else 0.0,
                    "frame_indices": indices,
                    "embedding": seg_emb.tolist(),
                }
            )
            prev = bp

        merged = merge_segments(raw_segments, embeddings, self.merge_sim_threshold)
        logger.info("Final segment count after merging: %d", len(merged))

        # Re-compute embeddings after merging
        for seg in merged:
            idxs = seg["frame_indices"]
            if idxs:
                m = embeddings[idxs].mean(axis=0)
                norm = np.linalg.norm(m)
                seg["embedding"] = (m / norm if norm > 1e-10 else m).tolist()

        return merged
