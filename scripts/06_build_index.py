"""Script 06 — Build a retrieval index from segment embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from vlm_video.common.io_jsonl import read_jsonl
from vlm_video.common.logging_utils import get_logger
from vlm_video.retrieval.index_factory import get_index

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a retrieval index from segment embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="Path to YAML config file")
    p.add_argument(
        "--segments_jsonl",
        required=True,
        help="Path to segments_pred.jsonl (from script 05)",
    )
    p.add_argument(
        "--out_dir",
        default="data/interim",
        help="Base output directory; index goes to <out_dir>/<video_id>/index/",
    )
    p.add_argument("--video_id", required=True, help="Identifier for this video")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from vlm_video.common.config import load_config

    cfg = load_config(args.config)
    backend = cfg["retrieval"]["backend"]

    seg_path = Path(args.segments_jsonl)
    if not seg_path.exists():
        logger.error("Segments file not found: %s", seg_path)
        return

    segments: list[dict[str, Any]] = list(read_jsonl(seg_path))
    if not segments:
        logger.error("No segments found in %s", seg_path)
        return

    # Extract embeddings array and strip it from metadata
    emb_list: list[np.ndarray] = []
    meta_list: list[dict[str, Any]] = []
    for i, seg in enumerate(segments):
        emb = seg.get("embedding")
        if emb is None:
            logger.warning("Segment %d has no embedding; skipping.", i)
            continue
        emb_list.append(np.array(emb, dtype=np.float32))
        meta = {k: v for k, v in seg.items() if k != "embedding"}
        meta["segment_id"] = i
        meta_list.append(meta)

    if not emb_list:
        logger.error("No valid embeddings found in segments file.")
        return

    embeddings = np.stack(emb_list, axis=0)
    logger.info("Building %s index from %s segments …", backend, len(emb_list))

    index = get_index(backend)
    index.build(embeddings, meta_list)

    index_dir = Path(args.out_dir) / args.video_id / "index"
    index.save(index_dir)
    logger.info("Index saved to %s", index_dir)
    print(f"Built {backend} index ({len(emb_list)} segments) → {index_dir}")


if __name__ == "__main__":
    main()
