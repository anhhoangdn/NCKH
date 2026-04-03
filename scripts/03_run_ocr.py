"""Script 03 — Optional OCR on extracted frames."""

from __future__ import annotations

import argparse
from pathlib import Path

from vlm_video.common.io_jsonl import write_jsonl
from vlm_video.common.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Tesseract OCR on extracted video frames (optional).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="Path to YAML config file")
    p.add_argument("--frames_dir", required=True, help="Directory containing frame images")
    p.add_argument(
        "--out_dir",
        default="data/interim",
        help="Base output directory; output goes to <out_dir>/<video_id>/",
    )
    p.add_argument("--video_id", required=True, help="Identifier for this video")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from vlm_video.common.config import load_config

    cfg = load_config(args.config)
    ocr_cfg = cfg.get("ocr", {})

    if not ocr_cfg.get("enabled", False):
        logger.info("OCR is disabled in config (ocr.enabled=false). Skipping.")
        print("OCR disabled. Set ocr.enabled: true in your config to enable.")
        return

    # Try to import the optional dependency
    try:
        from vlm_video.preprocess.ocr_wrapper import TesseractOCR
    except ImportError as exc:
        logger.warning("OCR dependencies not available: %s", exc)
        print(f"WARNING: {exc}")
        print("Skipping OCR. Install with: pip install vlm_video[ocr]")
        return

    frames_dir = Path(args.frames_dir)
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))

    if not frame_files:
        logger.warning("No frame files found in %s", frames_dir)
        return

    ocr = TesseractOCR(
        lang=ocr_cfg.get("lang", "vie+eng"),
        psm=ocr_cfg.get("psm", 3),
    )

    records = []
    for i, fp in enumerate(frame_files):
        text = ocr.extract_text(fp)
        records.append(
            {
                "video_id": args.video_id,
                "frame_idx": i,
                "frame_path": str(fp),
                "ocr_text": text,
            }
        )
        if (i + 1) % 50 == 0:
            logger.info("OCR progress: %d / %d frames", i + 1, len(frame_files))

    vid_dir = Path(args.out_dir) / args.video_id
    vid_dir.mkdir(parents=True, exist_ok=True)
    out_path = vid_dir / "ocr_results.jsonl"
    n = write_jsonl(out_path, records)
    logger.info("Saved %d OCR records to %s", n, out_path)
    print(f"OCR complete: {n} frames → {out_path}")


if __name__ == "__main__":
    main()
