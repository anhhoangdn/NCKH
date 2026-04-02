"""Script 05 — Segment the video using pre-computed embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vlm_video.common.io_jsonl import read_jsonl, write_jsonl
from vlm_video.common.logging_utils import get_logger
from vlm_video.segmentation.segmenter import VideoSegmenter

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Segment a video using pre-computed frame embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="Path to YAML config file")
    p.add_argument(
        "--embeddings_npz", required=True, help="Path to embeddings.npz (from script 04)"
    )
    p.add_argument(
        "--embedding_meta_jsonl",
        default=None,
        help="Path to embedding_meta.jsonl; if omitted, inferred from embeddings_npz location",
    )
    p.add_argument(
        "--out_dir",
        default="data/interim",
        help="Base output directory; outputs go to <out_dir>/<video_id>/",
    )
    p.add_argument("--video_id", required=True, help="Identifier for this video")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from vlm_video.common.config import load_config

    cfg = load_config(args.config)

    # Load embeddings
    npz_path = Path(args.embeddings_npz)
    if not npz_path.exists():
        logger.error("Embeddings file not found: %s", npz_path)
        return

    data = np.load(npz_path)
    embeddings: np.ndarray = data["embeddings"]
    logger.info("Loaded embeddings %s from %s", embeddings.shape, npz_path)

    # Load metadata for timestamps
    meta_path = (
        Path(args.embedding_meta_jsonl)
        if args.embedding_meta_jsonl
        else npz_path.parent / "embedding_meta.jsonl"
    )
    meta_records = list(read_jsonl(meta_path)) if meta_path.exists() else []
    timestamps = [r.get("timestamp_sec", i * (1 / cfg["frame_extraction"]["fps"]))
                  for i, r in enumerate(meta_records)]
    if not timestamps:
        fps = cfg["frame_extraction"]["fps"]
        timestamps = [i / fps for i in range(len(embeddings))]

    frame_paths = [r.get("frame_path", "") for r in meta_records]
    if not frame_paths:
        frame_paths = [""] * len(embeddings)

    # Run segmentation
    segmenter = VideoSegmenter(cfg)
    segments = segmenter.segment(
        frames=frame_paths,
        timestamps=timestamps,
        embeddings=embeddings,
    )

    # Add video_id
    for seg in segments:
        seg["video_id"] = args.video_id

    vid_dir = Path(args.out_dir) / args.video_id
    vid_dir.mkdir(parents=True, exist_ok=True)
    out_path = vid_dir / "segments_pred.jsonl"
    n = write_jsonl(out_path, segments)
    logger.info("Saved %d segments to %s", n, out_path)
    print(f"Segmentation: {n} segments → {out_path}")


if __name__ == "__main__":
    main()
