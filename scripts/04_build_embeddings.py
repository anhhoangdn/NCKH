"""Script 04 — Build per-frame CLIP embeddings with late fusion."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from vlm_video.common.io_jsonl import read_jsonl, write_jsonl
from vlm_video.common.logging_utils import get_logger
from vlm_video.embeddings.clip_encoder import CLIPEncoder
from vlm_video.embeddings.fusion import late_fusion

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute CLIP embeddings (visual + text + optional OCR) for each frame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="Path to YAML config file")
    p.add_argument("--frames_dir", required=True, help="Directory containing frame JPEGs")
    p.add_argument(
        "--transcript_jsonl", default=None, help="Path to transcript.jsonl (from script 02)"
    )
    p.add_argument(
        "--ocr_jsonl", default=None, help="Path to ocr_results.jsonl (from script 03)"
    )
    p.add_argument(
        "--out_dir",
        default="data/interim",
        help="Base output directory; outputs go to <out_dir>/<video_id>/",
    )
    p.add_argument("--video_id", required=True, help="Identifier for this video")
    return p.parse_args()


def load_transcripts(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        logger.warning("Transcript file not found: %s", p)
        return []
    return list(read_jsonl(p))


def load_ocr(path: str | Path | None) -> dict[int, str]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        logger.warning("OCR file not found: %s", p)
        return {}
    return {r["frame_idx"]: r.get("ocr_text", "") for r in read_jsonl(p)}


def text_at(timestamp: float, transcripts: list[dict[str, Any]]) -> str:
    for seg in transcripts:
        if seg.get("start", 0) <= timestamp <= seg.get("end", 0):
            return seg.get("text", "")
    return ""


def main() -> None:
    args = parse_args()

    from vlm_video.common.config import load_config

    cfg = load_config(args.config)
    emb_cfg = cfg["embeddings"]
    fps = cfg["frame_extraction"]["fps"]

    frames_dir = Path(args.frames_dir)
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        logger.error("No frames found in %s", frames_dir)
        return

    transcripts = load_transcripts(args.transcript_jsonl)
    ocr_map = load_ocr(args.ocr_jsonl)
    w = emb_cfg["weights"]
    wv, wt, wo = w["visual"], w["text"], w["ocr"]

    encoder = CLIPEncoder(
        model_name=emb_cfg["model"],
        pretrained=emb_cfg["pretrained"],
        device=emb_cfg.get("device", "cpu"),
    )

    emb_list: list[np.ndarray] = []
    meta_list: list[dict[str, Any]] = []

    for i, fp in enumerate(tqdm(frame_files, desc="Encoding frames")):
        t = i / fps
        vis = encoder.encode_image(fp)

        txt_emb = None
        txt = text_at(t, transcripts)
        if txt and wt > 0:
            txt_emb = encoder.encode_text(txt)

        ocr_emb = None
        ocr_text = ocr_map.get(i, "")
        if ocr_text and wo > 0:
            ocr_emb = encoder.encode_text(ocr_text)

        fused = late_fusion(vis, txt_emb, ocr_emb, wv, wt, wo)
        emb_list.append(fused)
        meta_list.append(
            {
                "video_id": args.video_id,
                "frame_idx": i,
                "frame_path": str(fp),
                "timestamp_sec": round(t, 3),
                "asr_text": txt,
                "ocr_text": ocr_text,
            }
        )

    embeddings = np.stack(emb_list, axis=0)
    vid_dir = Path(args.out_dir) / args.video_id
    vid_dir.mkdir(parents=True, exist_ok=True)

    emb_path = vid_dir / "embeddings.npz"
    np.savez_compressed(emb_path, embeddings=embeddings)
    logger.info("Saved embeddings %s to %s", embeddings.shape, emb_path)

    meta_path = vid_dir / "embedding_meta.jsonl"
    write_jsonl(meta_path, meta_list)
    logger.info("Saved metadata to %s", meta_path)
    print(f"Embeddings {embeddings.shape} → {emb_path}")
    print(f"Metadata   → {meta_path}")


if __name__ == "__main__":
    main()
