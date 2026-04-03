"""Script 07 — Retrieve relevant segments for a text query."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm_video.common.io_jsonl import write_jsonl
from vlm_video.common.logging_utils import get_logger
from vlm_video.common.timestamp import format_timestamp
from vlm_video.embeddings.text_encoder import TextEncoder
from vlm_video.retrieval.index_factory import get_index
from vlm_video.retrieval.ranking import rerank_results

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Encode a text query and retrieve the most relevant video segments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="Path to YAML config file")
    p.add_argument(
        "--index_dir", required=True, help="Directory containing the retrieval index"
    )
    p.add_argument("--query", required=True, help="Free-text retrieval query")
    p.add_argument("--top_k", type=int, default=None, help="Number of results to return")
    p.add_argument(
        "--video_id", default=None, help="Optional: filter results to a specific video"
    )
    p.add_argument(
        "--out_file",
        default=None,
        help="If set, save retrieval results to this JSONL path",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from vlm_video.common.config import load_config

    cfg = load_config(args.config)
    emb_cfg = cfg["embeddings"]
    ret_cfg = cfg["retrieval"]
    top_k = args.top_k if args.top_k is not None else ret_cfg.get("top_k", 5)
    backend = ret_cfg.get("backend", "sklearn")

    # Encode query
    encoder = TextEncoder(
        model_name=emb_cfg["model"],
        pretrained=emb_cfg["pretrained"],
        device=emb_cfg.get("device", "cpu"),
    )
    logger.info("Encoding query: %r", args.query)
    query_emb = encoder.encode(args.query)

    # Load index
    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        logger.error("Index directory not found: %s", index_dir)
        return

    index = get_index(backend)
    index.load(index_dir)
    logger.info("Loaded index with %d entries.", len(index))

    # Search
    results = index.search(query_emb, top_k=top_k)

    # Optional video_id filter
    if args.video_id:
        results = [r for r in results if r.get("video_id") == args.video_id]

    results = rerank_results(results, query_text=args.query)

    # Pretty-print results
    print(f"\nQuery: {args.query!r}")
    print(f"Top {len(results)} results:\n")
    for rank, res in enumerate(results, start=1):
        start = res.get("start_time", 0.0)
        end = res.get("end_time", 0.0)
        score = res.get("score", 0.0)
        vid = res.get("video_id", "?")
        print(
            f"  [{rank}] score={score:.4f}  "
            f"{format_timestamp(start)} → {format_timestamp(end)}  "
            f"video={vid}"
        )

    # Optionally save
    if args.out_file:
        out_path = Path(args.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        records = [{"query": args.query, "rank": i + 1, **r} for i, r in enumerate(results)]
        write_jsonl(out_path, records)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
