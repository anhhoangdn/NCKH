"""Script 01 — Extract frames from a video using ffmpeg."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vlm_video.common.io_jsonl import write_jsonl
from vlm_video.common.logging_utils import get_logger
from vlm_video.preprocess.ffmpeg_utils import check_ffmpeg, extract_frames

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract frames from a video at a fixed frame rate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="Path to YAML config file")
    p.add_argument("--input_video", required=True, help="Path to the source video")
    p.add_argument(
        "--out_dir",
        default="data/interim",
        help="Base output directory; frames go to <out_dir>/<video_id>/frames/",
    )
    p.add_argument("--video_id", required=True, help="Identifier for this video (no spaces)")
    p.add_argument("--fps", type=float, default=None, help="Override frame rate from config")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load config for fps default
    from vlm_video.common.config import load_config

    cfg = load_config(args.config)
    fps = args.fps if args.fps is not None else cfg["frame_extraction"]["fps"]

    if not check_ffmpeg():
        sys.exit(1)

    video_path = Path(args.input_video)
    if not video_path.exists():
        logger.error("Video file not found: %s", video_path)
        sys.exit(1)

    frames_dir = Path(args.out_dir) / args.video_id / "frames"
    frame_paths = extract_frames(video_path, frames_dir, fps=fps)

    # Build and save manifest
    manifest = [
        {
            "video_id": args.video_id,
            "frame_idx": i,
            "path": p,
            "timestamp_sec": round(i / fps, 3),
        }
        for i, p in enumerate(frame_paths)
    ]
    manifest_path = Path(args.out_dir) / args.video_id / "frame_manifest.jsonl"
    n = write_jsonl(manifest_path, manifest)
    logger.info("Saved %d frame records to %s", n, manifest_path)
    print(f"Extracted {len(frame_paths)} frames → {frames_dir}")
    print(f"Manifest  → {manifest_path}")


if __name__ == "__main__":
    main()
